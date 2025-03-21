import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache
from sentence_transformers import SentenceTransformer

def continue_from_system(user_content, zs_cot = True):
    out = f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if zs_cot: out = out + "Let's think step by step. "
    return out
    
class MA_LLM_hf():
    def __init__(self, num_agents, max_new_tokens = 30, max_context_length = 200, device = 'auto'): # if device == auto, please change conversions below.
        self.num_agents = num_agents
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.device = device
        # Initalization of LLM + sentence embedding model for RAG (using instruct llama3.2-3b)
        self.TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", padding_side='left')
        self.LLM_MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=th.float16, #th.bfloat16
                                                              quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=device)
        self.TOKENIZER.pad_token = self.TOKENIZER.eos_token
        self.LLM_MODEL.generation_config.pad_token_id = self.TOKENIZER.pad_token_id
        self.SENTENCE_EMBED = SentenceTransformer("all-MiniLM-L6-v2").to(device) # embedding size = 384
        self.cosine_similarity = th.nn.CosineSimilarity()
        self.reset()

    def reset(self):
        # Resets context memory
        self.vector_memory = [[] for _ in range(self.num_agents)]
        self.text_memory = [[] for _ in range(self.num_agents)]

    def retrieve(self, embedding):
        # Creates system prompt/cache and retrieves context based on cosine-similiarity score
        system_prompt = []
        for agent in range(self.num_agents):
            if len(self.vector_memory[agent]) == 0:
                agent_system_prompt = f"You are agent {agent} of {self.num_agents} agents. You are starting out on a multi-agent task."
            else:
                agent_system_prompt = f"You are agent {agent} of {self.num_agents} agents in a multi-agent task. Use the following relevant context:\n"
                agent_embedding = embedding[:,agent]
                score = th.stack([self.cosine_similarity(agent_embedding[i].squeeze(0).to(self.device), self.vector_memory[agent].to(self.device))
                                  for i in range(embedding.shape[0])], 0)
                indices = score.argmax(-1)
                num_context = 0
                length = 0
                for i in indices:
                    context = self.text_memory[agent][i]
                    agent_system_prompt += f"Context {num_context}:\n{context}\n"
                    length += len(context)
                    num_context += 1
                    if length > self.max_context_length:
                        break
            system_prompt.append(agent_system_prompt)
        return system_prompt

    @th.no_grad()
    def get_communication(self, observation, system_prompts):
        # From observation, computes message to be broadcast to all other agents. This message is prepended to the user content.
        all_prompts = []
        batch = observation.shape[0]
        for agent in range(self.num_agents):
            for obs in observation:
                all_prompts.append(system_prompts[agent] \
                                   + continue_from_system(f"Your state is {obs[agent].cpu().tolist()}. Communicate crucial information to all other agents in a concise manner to accomplish the task."))
        model_inputs = self.TOKENIZER(all_prompts, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.LLM_MODEL.generate(**model_inputs, max_new_tokens=self.max_new_tokens) 
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        output = self.TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        return [output[agent * batch:(agent + 1) * batch] for agent in range(self.num_agents)]

    @th.no_grad()
    def make_decision_embedding(self, observation, communication, system_prompts):
        """
        Construct the user message (to be prompted to LLM) using observation + communication that is then prompted to the LLM.
        The output text (i.e. the assistant content) is returned.
        Output: Tuple(embedding vector of output text (using sentence embedding), output text)
        """
        all_prompts = []
        batch = observation.shape[0]
        for agent in range(self.num_agents):
            for b, obs in enumerate(observation):
                user_prompt_content = ""
                if communication != None:
                    user_prompt_content += f"Here is what other agents have told you:\n"
                    for other_agent, com in enumerate(communication):
                        if agent != other_agent:
                            user_prompt_content += f"Agent {other_agent}: {com[b]}\n"
                user_prompt_content += f"\nAnalyze the current state: {obs[agent].tolist()} to make your next decision."
                all_prompts.append(system_prompts[agent] + continue_from_system(user_prompt_content))
        model_inputs = self.TOKENIZER(all_prompts, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.LLM_MODEL.generate(**model_inputs, max_new_tokens=self.max_new_tokens) 
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        output = self.TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        embedding = self.SENTENCE_EMBED.encode(output, show_progress_bar =False)
        return (th.stack([th.tensor(embedding[agent * batch:(agent + 1) * batch]) for agent in range(self.num_agents)], 1).to(self.device), 
                [output[agent * batch:(agent + 1) * batch] for agent in range(self.num_agents)])
    
    @th.no_grad()
    def update_memory(self, observation, action, reward, system_prompts):
        # Using the observation, action and reward recieved, the agent self-reflects and appends to the context memory.
        all_prompts = []
        batch = observation.shape[0]
        for agent in range(self.num_agents):
            for obs, act, rew in zip(observation, action, reward):
                all_prompts.append(system_prompts[agent] \
                                   + continue_from_system(f"Your state is {obs[agent].tolist()}, the action you chose was {act[agent].tolist()}, and the reward you recieved was {rew[agent].tolist()}. Given the information above, retain the most salient highlevel concepts for future decision-making.”"))
        model_inputs = self.TOKENIZER(all_prompts, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.LLM_MODEL.generate(**model_inputs, max_new_tokens=self.max_new_tokens) 
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        output = self.TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        embedding = self.SENTENCE_EMBED.encode(output, show_progress_bar =False)
        for agent in range(self.num_agents):
            if len(self.vector_memory[agent]) == 0:
                self.vector_memory[agent] = th.tensor(embedding[agent * batch:(agent + 1) * batch])
            else:
                self.vector_memory[agent] = th.cat([self.vector_memory[agent], th.tensor(embedding[agent * batch:(agent + 1) * batch])], 0)
            self.text_memory[agent].extend(output[agent * batch:(agent + 1) * batch])

            # Limit memory to last 100000 contexts to limit ram usage (memory is shared between batches. need some change im future.)
            # self.vector_memory[agent] = self.vector_memory[agent][-100000:]
            # self.text_memory[agent] = self.text_memory[agent][-100000:]
    

# 
# from openai import OpenAI
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"), max_retries=5)



