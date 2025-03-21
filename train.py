import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import tqdm
from moviepy.editor import ImageSequenceClip

import numpy as np
import torch as th
# from torch.utils.tensorboard import SummaryWriter
from agent import Agent
import vmas
import logging, time, random
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configs/code were adapted and modified from BenchRL: https://github.com/facebookresearch/BenchMARL and CleanRL: https://github.com/vwxyzjn/cleanrl
@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    save_folder = Path(HydraConfig.get().runtime.output_dir)
    task_name = hydra_choices.task
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))
    print(f"Saving in: {save_folder}")

    device = cfg.experiment.device
    num_steps = cfg.experiment.num_steps_data_collection
    num_agents = cfg.task.n_agents
    num_envs = cfg.experiment.num_envs

    # Initialize environments
    envs = vmas.make_env(scenario = task_name.split('/')[-1],
                         num_envs = num_envs,
                         device = device,
                         **cfg.task)
    test_envs = vmas.make_env(scenario = task_name.split('/')[-1],
                         num_envs = 1,
                         device = device,
                         seed = cfg.seed,
                         **cfg.task)
    observation_space = {k:v.shape[0] for k,v in envs.observation_space.sample().items()}
    action_space = {k:v.shape[0] for k,v in envs.action_space.sample().items()}
    obs_dim = max(observation_space.values())
    act_dim = max(action_space.values())
    
    # Initialize model + optimizer
    agent = Agent(observation_space, action_space,
                  device = device,
                  llm_device = cfg.experiment.llm_device,
                  max_new_tokens = cfg.experiment.max_new_tokens,
                  max_context_length = cfg.experiment.max_context_length)
    optimizer = th.optim.Adam(agent.parameters(), lr=cfg.experiment.lr, eps=cfg.experiment.eps)

    # Initialize storage for data collection
    obs = th.zeros((num_steps, num_envs, num_agents, obs_dim)).to(device)
    actions = th.zeros((num_steps, num_envs, num_agents, act_dim)).to(device)
    logprobs = th.zeros((num_steps, num_envs, num_agents)).to(device)
    rewards = th.zeros((num_steps, num_envs, num_agents)).to(device)
    dones = th.zeros((num_steps, num_envs, 1)).to(device)
    values = th.zeros((num_steps, num_envs, num_agents)).to(device)

    # Set up the context memory
    agent.llm.reset()
    for iteration in range(cfg.experiment.num_iterations):

        # ---------------- #
        # Data collection  #
        # ---------------- #
        logger.info(f"Iteration {iteration} / {cfg.experiment.num_iterations}: Data Collection...")
        next_obs = envs.reset()
        next_done = th.zeros(num_envs, 1).to(device)
        for step in tqdm.tqdm(range(num_steps)):
            update_llm_memory = not step % cfg.experiment.update_llm_memory
            communicate = not step % cfg.experiment.communicate
            with th.no_grad():
                next_obs = th.stack([next_obs[k] for k in agent.agent_keys],1)
                action, logprob, _, value, system_prompt, context_embedding = agent.get_action_and_value(next_obs, communicate = communicate)
                values[step] = value
            obs[step] = next_obs
            dones[step] = next_done.view(-1, 1)
            actions[step] = action
            logprobs[step] = logprob
            action_dict = {k: action[:,i] for i,k in enumerate(agent.agent_keys)}
            next_obs, reward, next_done, infos = envs.step(action_dict)
            rewards[step] = th.stack([reward[k] for k in agent.agent_keys], 1)
            if update_llm_memory:
                agent.llm.update_memory(obs[step], action, 
                                        th.stack([reward[k] for k in agent.agent_keys], 1), 
                                        system_prompt)
        # Compute advantage and return for all samples
        with th.no_grad():
            next_obs = th.stack([next_obs[k] for k in agent.agent_keys],1)
            next_value = agent.get_value(next_obs)
            advantages = th.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float().view(-1, 1)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.algorithm.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg.algorithm.gamma * cfg.algorithm.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # ---------------- #
        # PPO Updates      #
        # ---------------- #
        logger.info(f"Iteration {iteration} / {cfg.experiment.num_iterations}: Data Collection Completed")
        logger.info(f"Iteration {iteration} / {cfg.experiment.num_iterations}: RL Update...")

        b_obs = obs.reshape((-1, num_agents, obs_dim))
        b_logprobs = logprobs.reshape(-1, num_agents)
        b_actions = actions.reshape((-1, num_agents, act_dim))
        b_advantages = advantages.reshape(-1, num_agents)
        b_returns = returns.reshape(-1, num_agents)
        b_values = values.reshape(-1, num_agents)

        b_inds = np.arange(b_obs.shape[0])
        clipfracs = []
        for epoch in tqdm.tqdm(range(cfg.experiment.num_update_epochs)):
            np.random.shuffle(b_inds)
            n = b_obs.shape[0] // cfg.experiment.train_batch_size
            for start in range(0, n, cfg.experiment.train_batch_size):
                end = start + cfg.experiment.train_batch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], communicate = True)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with th.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg.algorithm.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if cfg.algorithm.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Compute policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - cfg.algorithm.clip_coef, 1 + cfg.algorithm.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Compute value loss
                if cfg.algorithm.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.algorithm.clip_coef,
                        cfg.algorithm.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.algorithm.ent_coef * entropy_loss + v_loss * cfg.algorithm.vf_coef
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(agent.parameters(), cfg.experiment.max_grad_norm)
                optimizer.step()
                if cfg.algorithm.target_kl is not None and approx_kl > cfg.algorithm.target_kl:
                    break

        # ---------------- #
        # Evaluation       #
        # ---------------- #
        if not iteration % cfg.experiment.eval_freq:
            logging.info(f"Iteration {iteration} / {cfg.experiment.num_iterations}: Evaluation...")
            frame_list = []
            _obs = test_envs.reset()
            frame = test_envs.render(mode="rgb_array", agent_index_focus=None)
            frame_list.append(frame)
            eval_score = {}
            step = 0
            while True:
                with th.no_grad():
                    communicate = not step % cfg.experiment.communicate
                    _obs = th.stack([_obs[k] for k in agent.agent_keys], 1)
                    _action = agent.get_action_and_value(_obs.to(device), communicate = communicate)[0]
                    _action = {k: _action[:,i] for i, k in enumerate(agent.agent_keys)}
                    _obs, _reward, _done, _ = test_envs.step(_action)
                    frame = test_envs.render(mode="rgb_array", agent_index_focus=None)
                    frame_list.append(frame)
                    for i, k in enumerate(agent.agent_keys):
                        if k not in eval_score:
                            eval_score[k] = _reward[k]
                        else:
                            eval_score[k] += _reward[k]
                    step += 1
                if _done.all():
                    break
            # Save video of evaluation.
            eval_score = {k: v.mean().item() for k,v in eval_score.items()}
            logging.info(f"Iteration {iteration}: Score [{sum(eval_score.values())}] {eval_score}")
            clip = ImageSequenceClip(frame_list, fps=30)
            clip.write_videofile(f'{save_folder}/{iteration}.mp4', fps=30)



if __name__ == "__main__":
    hydra_experiment()
