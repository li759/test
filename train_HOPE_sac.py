import sys
sys.path.append('..')
sys.path.append('.')
import time
import os
import pickle
# Force matplotlib Agg backend early to avoid Tkinter main loop issues
try:
    import matplotlib
    if matplotlib.get_backend().lower() != 'agg':
        matplotlib.use('Agg', force=True)
except Exception:
    pass
from shutil import copyfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from model.agent.sac_agent import SACAgent as SAC
from model.agent.parking_agent import ParkingAgent, RsPlanner
from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper
from env.vehicle import VALID_SPEED, Status
from evaluation.eval_utils import eval
from configs import *

class SceneChoose:

    def __init__(self) -> None:
        self.scene_types = {0: 'Normal', 1: 'Complex', 2: 'Extrem', 3: 'dlp'}
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []
        self.history_horizon = 200

    def choose_case(self):
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        elif np.random.random() > 0.5:
            scene_chosen = self._choose_case_worst_perform()
        else:
            scene_chosen = self._choose_case_uniform()
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]

    def update_success_record(self, success: int):
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i + 1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)

    def _choose_case_worst_perform(self):
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record) / len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate / np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

class DlpCaseChoose:

    def __init__(self) -> None:
        self.dlp_case_num = 248
        self.case_record = []
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500

    def choose_case(self):
        if np.random.random() < 0.2 or len(self.case_record) < self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record) / len(recent_success_record))
        fail_rate = 1 - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate / np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

    def update_success_record(self, success: int, case_id: int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)


def _sample_valid_action(env, valid_steers, valid_speed, action_mask):
    # 处理边界情况
    if len(valid_steers) == 0:
        return env.action_space.sample()

    # 确保转向角度有序（从小到大排序）
    sort_indices = np.argsort(valid_steers)
    valid_steers = valid_steers[sort_indices]
    valid_speed = valid_speed[sort_indices]

    # 采样动作，直到找到有效的
    for attempt in range(MAX_ACTION_SAMPLING_ATTEMPTS):
        action = env.action_space.sample()
        steer = action[0]
        speed = action[1]

        # 检查动作是否在有效范围内
        if _is_action_valid(steer, speed, valid_steers, valid_speed):
            return action

    # 如果多次尝试都失败，返回随机动作
    print(f"Warning: Failed to sample valid action after {MAX_ACTION_SAMPLING_ATTEMPTS} attempts")
    return env.action_space.sample()


def _is_action_valid(steer, speed, valid_steers, valid_speed):
    # 处理只有一个有效转向的情况
    if len(valid_steers) == 1:
        steer_diff = abs(steer - valid_steers[0])
        speed_match = (speed > 0 and valid_speed[0] > 0) or (speed < 0 and valid_speed[0] < 0)
        return steer_diff < 0.1 and speed_match  # 0.1是转向容差阈值

    # 处理多个有效转向的情况
    for i in range(len(valid_steers) - 1):
        s1, s2 = valid_steers[i], valid_steers[i+1]

        # 检查转向是否在区间内
        steer_in_range = (min(s1, s2) <= steer <= max(s1, s2))

        # 检查速度方向是否匹配（同向）
        speed_direction_match = (speed * valid_speed[i] > 0) and (speed * valid_speed[i+1] > 0)

        if steer_in_range and speed_direction_match:
            return True

    return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#sys.argv = ['train_HOPE_sac.py', '--verbose', 'True', '--visualize', 'False']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None)
    parser.add_argument('--img_ckpt', type=str, default='./model/ckpt/autoencoder.pt')
    parser.add_argument('--train_episode', type=int, default=4000)
    parser.add_argument('--eval_episode', type=int, default=1)
    #parser.add_argument('--train_episode', type=int, default=100000)
    #parser.add_argument('--eval_episode', type=int, default=2000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    # 修改参数定义
    #parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()
    verbose = args.verbose
    if args.visualize:
        raw_env = CarParking(fps=100, verbose=verbose)
    else:
        raw_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    env = CarParkingWrapper(raw_env)
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime('%Y%m%d_%H%M%S', current_time)
    save_path = relative_path + '/log/exp/sac_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    copyfile('./configs.py', save_path + 'configs.txt')
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)
    seed = SEED
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {'discrete': False, 'observation_shape': env.observation_shape, 'action_dim': env.action_space.shape[0], 'hidden_size': 64, 'activation': 'tanh', 'dist_type': 'gaussian', 'save_params': False, 'actor_layers': actor_params, 'critic_layers': critic_params}
    print('observation_space:', env.observation_space)
    rl_agent = SAC(configs)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')
    img_encoder_checkpoint = args.img_ckpt if USE_IMG else None
    if img_encoder_checkpoint is not None and os.path.exists(img_encoder_checkpoint):
        rl_agent.load_img_encoder(img_encoder_checkpoint, require_grad=UPDATE_IMG_ENCODE)
    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)
    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    total_step_num = 0
    best_success_rate = [0, 0, 0, 0]
    start_episode = 0
    
    # 如果加载了检查点，尝试恢复训练状态
    if checkpoint_path is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                training_state = pickle.load(f)
                start_episode = training_state.get('episode', 0)
                total_step_num = training_state.get('total_step_num', 0)
                reward_list = training_state.get('reward_list', [])
                succ_record = training_state.get('succ_record', [])
                best_success_rate = training_state.get('best_success_rate', [0, 0, 0, 0])
                print(f'Resumed training from episode {start_episode}, total steps: {total_step_num}')
    
    #print('_________________________________________________________________________________')
    for i in range(start_episode, args.train_episode):
        '''scene_chosen = scene_chooser.choose_case()
        if scene_chosen == 'dlp':
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None'''
        #Q
        case_id = None
        scene_chosen = "Complex"
        if len(scene_chooser.scene_record) == 0:
            # 第一次运行，手动添加Complex场景的索引
            scene_chooser.scene_record.append(1)  # Complex场景的索引是1
            scene_chooser.current_scene = 1
        #print('episode: ', args.train_episode)
        #print('scene: ', scene_chosen)
        obs = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        xy = []
        #print('1_________________________________________________________________________________')
        while not done:
            step_num += 1
            total_step_num += 1
            #print(not parking_agent.executing_rs)
            if total_step_num <= parking_agent.configs.memory_size and (not parking_agent.executing_rs):
                action_mask = obs['action_mask']
                valid_indices = np.where(action_mask > ACTION_MASK_THRESHOLD)[0]

                if len(valid_indices) == 0 or np.all(action_mask <= ACTION_MASK_THRESHOLD):
                    action = env.action_space.sample()
                else:
                    all_actions = np.array(env.action_filter.action_space)
                    valid_steers = all_actions[valid_indices, 0]
                    valid_speed = all_actions[valid_indices, 1]
                    action = _sample_valid_action(env, valid_steers, valid_speed, action_mask)

                log_prob = parking_agent.get_log_prob(obs, action)
            else:
                action, log_prob = parking_agent.get_action(obs)
            #print("action: ", action)
            next_obs, reward, done, info = env.step(action)
            reward_info.append(list(info['reward_info'].values()))
            total_reward += reward
            reward_per_state_list.append(reward)
            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            #print("total_step_num: ", total_step_num)
            #print("parking_agent.configs.memory_size: ", parking_agent.configs.memory_size)
            #print("total_step_num > parking_agent.configs.memory_size: ", total_step_num > parking_agent.configs.memory_size)
            if total_step_num > parking_agent.configs.memory_size and total_step_num % 5 == 0:
                actor_loss, critic_loss = parking_agent.update()
                if total_step_num % 200 == 0:
                    writer.add_scalar('actor_loss', actor_loss, i)
                    writer.add_scalar('critic_loss', critic_loss, i)
            if info['path_to_dest'] is not None:
                parking_agent.set_planner_path(info['path_to_dest'])
            if done:
                if info['status'] == Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(1, case_id)
                    print("ARRIVED--- ", "case_id: ", case_id, "scene_chosen: ",  scene_chosen, "reward: ", total_reward)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == 'dlp':
                        dlp_case_chooser.update_success_record(0, case_id)
                    print("NOT ARRIVED--- ", "case_id: ", case_id, "scene_chosen: ", scene_chosen, "reward: ", total_reward)
        #print('2_________________________________________________________________________________')
        writer.add_scalar('total_reward', total_reward, i)
        writer.add_scalar('avg_reward', np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar('action_std0', parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0], i)
        writer.add_scalar('action_std1', parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1], i)
        writer.add_scalar('alpha', parking_agent.alpha.detach().cpu().numpy().reshape(-1)[0], i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar('success_rate_%s' % scene_chooser.scene_types[type_id], np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar('step_num', step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info, 2)
        reward_info_list.append(list(reward_info))
        if verbose and i % 10 == 0 and (i > 0):
            print('success rate:', np.sum(succ_record), '/', len(succ_record))
            print(parking_agent.log_std.detach().cpu().numpy().reshape(-1), parking_agent.alpha.detach().cpu().numpy().reshape(-1))
            print('episode:%s  average reward:%s' % (i, np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]), np.mean(parking_agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            #for j in range(10):
                #print(case_id_list[-(10 - j)], reward_list[-(10 - j)], reward_info_list[-(10 - j)])
            print('')
        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
            success_rate_dlp = np.mean(scene_chooser.success_record[3][-100:])
        if success_rate_normal >= best_success_rate[0] and success_rate_complex >= best_success_rate[1] and (success_rate_extreme >= best_success_rate[2]) and (success_rate_dlp >= best_success_rate[3]) and (i > 100):
            raw_best_success_rate = np.array([success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp])
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save('%s/SAC1_best.pt' % save_path, params_only=True)
            # 保存最佳模型时的训练状态
            training_state = {
                'episode': i + 1,
                'total_step_num': total_step_num,
                'reward_list': reward_list,
                'succ_record': succ_record,
                'best_success_rate': best_success_rate
            }
            with open('%s/training_state.pkl' % save_path, 'wb') as f:
                pickle.dump(training_state, f)
            f_best_log = open(save_path + 'best.txt', 'w')
            f_best_log.write('epoch: %s, success rate: %s %s %s %s' % (i + 1, raw_best_success_rate[0], raw_best_success_rate[1], raw_best_success_rate[2], raw_best_success_rate[3]))
            f_best_log.close()
        if (i + 1) % 2000 == 0:
            parking_agent.save('%s/SAC1_%s.pt' % (save_path, i), params_only=True)
            # 保存训练状态
            training_state = {
                'episode': i + 1,
                'total_step_num': total_step_num,
                'reward_list': reward_list,
                'succ_record': succ_record,
                'best_success_rate': best_success_rate
            }
            with open('%s/training_state.pkl' % save_path, 'wb') as f:
                pickle.dump(training_state, f)
        if verbose and i % 20 == 0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0, j - 50):j + 1]) for j in range(len(reward_list))]
            plt.plot(episodes, reward_list)
            plt.plot(episodes, mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            f = plt.gcf()
            f.savefig('%s/reward.png' % save_path)
            f.clear()
        #print('2_________________________________________________________________________________')
    eval_episode = args.eval_episode
    choose_action = False
    print('eval_episode: ',eval_episode)
    '''with torch.no_grad():
        env.set_level('dlp')
        log_path = save_path + '/dlp'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        print("eval_dlp")
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        env.set_level('Extrem')
        log_path = save_path + '/extreme'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        print("eval_extreme")
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        env.set_level('Complex')
        log_path = save_path + '/complex'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        print("eval_complex")
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
        env.set_level('Normal')
        log_path = save_path + '/normalize'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        print("eval_normalize")
        eval(env, parking_agent, episode=eval_episode, log_path=log_path, post_proc_action=choose_action)
    env.close()'''