import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agents.rnn_agent import RNNAgent
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from robotic_warehouse.human_play import InteractiveRWAREEnv
from argparse import ArgumentParser


class TrainedAgent:
    def __init__(self, path):
        self.path = path
        self.model = None
    
    def load_agent(self):
        if self.model is None: 
            self.model = RNNAgent(75, True)
            state_dict = th.load(self.path)
            self.model.load_state_dict(state_dict)
        return self.model

    def get_action(self, agent_inputs):
        with th.no_grad():
            agent_inputs_tensor = th.tensor(agent_inputs, dtype=th.float32)
            outputs = self.model(agent_inputs_tensor)
            
            #agent_outs = trained_agent(agent_inputs, self.hidden_states)[0]
            #agent_outs = agent_outs.clone().detach().requires_grad_(True)
        return outputs


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args, help_flag=True, trained_agent_path=None):
        self.n_agents = args.n_agents
        self.args = args
        self.help_flag = help_flag
        #self.k_steps = args.k_steps  
        self.k_steps = 10
        self.step_counter = 1  

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

        self.interactive_env = InteractiveRWAREEnv(env="rware-tiny-4ag-v2", 
                                                   #max_steps=args.max_steps,   # CHANGE
                                                   max_steps=500,
                                                   display_info=True) # CHANGE


    def select_actions(self, ep_batch, t_ep, t_env, test_mode=False):
        if self.help_flag and self.step_counter % self.k_steps == 0:  # human input step
            obss, actions = self.interactive_env.get_current_human_action()
            actions = [act.value for act in actions]
        else:  # agent policy step
            path = os.path.expanduser("~/PERSONAL-DIR/UOA-NEW/human-rware/results/models/mappo_seed663242369_rware:rware-tiny-4ag-v2_2024-11-02 16:24:42.284341/5000500/agent.th")
            trained_agent = TrainedAgent(path).load_agent()
            print("Successfuly loaded the trained agent")
            
            agent_inputs = self._build_inputs(ep_batch, t_ep)
            action = trained_agent.get_action(agent_inputs)
            
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            agent_outputs = self.forward(agent_inputs, avail_actions, ep_batch, t_ep, test_mode=test_mode)
            actions = self.action_selector.select_action(agent_outputs, avail_actions, t_env, test_mode=test_mode)
        print(f'step_counter: {self.step_counter}')
        self.step_counter += 1 
        return actions

    def forward(self, agent_inputs, ep_batch, avail_actions, t, test_mode=False):
        #agent_inputs = self._build_inputs(ep_batch, t)
        #avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
