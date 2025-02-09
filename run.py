import os
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

import gym_super_mario_bros                                      #導入gym_super_mario_bros，這是一個基於 Gym 的模組，用於模擬《Super Mario Bros》遊戲環境。
from nes_py.wrappers import JoypadSpace                          #從nes_py中導入JoypadSpace，用於限制遊戲中可用的按鈕動作（例如僅允許「移動右」或「跳躍」的動作集合）。
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT         #從 gym_super_mario_bros中導入SIMPLE_MOVEMENT，這是一個預定義的按鈕動作集合（如「右移」、「跳躍」等），用於控制 Mario 的行為。
                                                                 #簡化動作空間 NES 控制器有 8 個按鍵（上下左右、A、B、Select、Start），可能的按鍵組合數非常大

from utils import preprocess_frame                               #用於對遊戲的畫面進行預處理，例如灰階化、調整大小等，將其轉換為適合神經網路輸入的格式
from reward import *                                             #模組中導入所有函式，這些函式用於設計和計算自定義獎勵（例如根據 Mario 的硬幣數量、水平位移等來計算獎勵）。
from model import CustomCNN                                      #自定義的卷積神經網路模型，用於處理遊戲畫面並生成動作決策
from DQN import DQN, ReplayMemory                                #用於執行強化學習的主要邏輯 DQN模組中導入回放記憶體，用於存儲和抽取遊戲的狀態、動作、獎勵等樣本，提升訓練穩定性。



# ========== config ===========
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#========= basic train config==============================================
LR = 0.00001
BATCH_SIZE = 32                 #達到batch size更新主網路參數 達到50次更新目標網路的參數
GAMMA = 0.99                    #控制模型對長期獎勵和短期獎勵的權衡 gamma靠近1 模型更重視長期獎勵
MEMORY_SIZE = 10000             #用來儲存，遊戲過程中的記錄 如果存超過了 會刪除最早進來的
EPSILON_START = 0.3             #初始探索率
EPSILON_END = 0.2               #在訓練過程中，會逐漸從探索（隨機選擇動作）轉向利用（選擇模型預測的最佳動作）。
                                #EPSILON的值會隨著訓練進展逐漸下降，直到達到此最小值0.3
                                #即訓練後期仍保留 30% 的探索概率，避免模型陷入局部最優解
TARGET_UPDATE = 50              #每隔幾回合去更新目標網路的權重
TOTAL_TIMESTEPS = 1000          #總訓練的回合數
VISUALIZE = True                #是否在訓練過程中渲染遊戲畫面 顯示遊戲畫面
MAX_STAGNATION_STEPS = 500       # Max steps without x_pos change 500

# 指數衰減相關參數
DECAY_FACTOR = 0.01  # 控制衰減速度（值越小，衰減越快）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA is available! Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")



# ========================DQN Initialization==========================================
obs_shape = (1, 84, 84)                         #obs_shape = (1, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)                #定義動作空間大小，使用SIMPLE_MOVEMENT中的動作數量（例如向右移動、跳躍等）
model = CustomCNN                               #指定模型架構為CustomCNN用於處理圖像並預測各動作的 Q 值
dqn = DQN(                                      #初始化 DQN agent
    model=model,
    state_dim=obs_shape,                        #狀態空間大小
    action_dim=n_actions,                       #動作空間大小
    learning_rate=LR,                           #學習率
    gamma=GAMMA,                                #折扣因子，用於計算未來獎勵
    epsilon_start=EPSILON_START,  # 初始探索率
    epsilon_end=EPSILON_END,      # 最小探索率
    epsilon_decay=DECAY_FACTOR,  # 衰減步幅                       #初始探索率
    target_update=TARGET_UPDATE,                #目標網路更新頻率
    device=device
)

# 自定義權重名稱與路徑
pretrained_weights_path = "./BEST/step_16_reward_25026_custom_25026.pth"  # 修改為你的權重路徑

# 加載預訓練的權重
if os.path.exists(pretrained_weights_path):
    print(f"Loading pretrained weights from {pretrained_weights_path}...")
    dqn.q_net.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    dqn.tgt_q_net.load_state_dict(dqn.q_net.state_dict())  # 同步目標網路
    print("Pretrained weights loaded successfully.")
else:
    print(f"No pretrained weights found at {pretrained_weights_path}. Training from scratch.")

'''
# 如果不想使用預訓練權重，直接從頭開始訓練
print("Training from scratch. No pretrained weights loaded.")
'''

memory = ReplayMemory(MEMORY_SIZE)              #創建經驗回放記憶體，用於存儲狀態轉移
step = 0                                        #記錄總步數
best_reward = -float('inf')                     # 儲存最佳累積獎勵Track the best reward in each SAVE_INTERVAL  
official_best_reward = -float('inf')      
cumulative_reward = 0                           # 當前時間步的總累積獎勵Track cumulative reward for the current timestep



#=======================訓練開始============================
for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress"):  #主訓練迴圈，進行TOTAL_TIMESTEPS次迭代
    state = env.reset()                                                         #重置遊戲環境，獲取初始狀態
    state = preprocess_frame(state)                                             #使用preprocess_frame 將畫面處理為灰階、縮放為84x84
    state = np.expand_dims(state, axis=0)                                       #新增一個維度，適配模型輸入

    done = False                                                                #表示當前遊戲是否結束
    prev_info = {                                                               #用於追蹤遊戲狀態（如水平位置、得分、硬幣數量）
        "x_pos": 0,  # Starting horizontal position (int).
        "y_pos": 0,  # Starting vertical position (int).
        "score": 0,  # Initial score is 0 (int).
        "coins": 0,  # Initial number of collected coins is 0 (int).
        "time": 400,  # Initial time in most levels of Super Mario Bros is 400 (int).
        "flag_get": False,  # Player has not yet reached the end flag (bool).
        "life": 3,  # Default initial number of lives is 3 (int).
    }

    cumulative_custom_reward = 0                                              #自定義獎勵總和
    cumulative_reward = 0 
    stagnation_time = 0                                                           #stagnation_time記錄遊戲角色在水平方向的停滯時間
    #開始一個回合的遊戲循環
    jump_streak = 0  # 初始化 jump_streak
    while not done:
        action = dqn.take_action(state)                                           #輸入目前狀態，交給DQN去做下一步
        next_state, reward, done, info = env.step(action)                         #執行動作並從環境中獲取下一狀態、回報、遊戲結束標記、以及遊戲資訊 
       
       
        # preprocess image state 將下一狀態進行預處理並調整為適合模型的形狀
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        cumulative_reward += reward   #更新累積獎勵

       # 使用指數衰減更新 epsilon
        dqn.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-DECAY_FACTOR * timestep)


        # 更新 jump_streak
        if "A" in SIMPLE_MOVEMENT[action]:
            jump_streak += 1
        else:
            jump_streak = 0  # 如果不是跳躍動作歸0

        # 打印 jump_streak
        #print(f"Timestep {timestep}, Jump Streak: {jump_streak}")

        # ===========================調用 reward.py 中的獎勵函數  包括硬幣獎勵、水平位移獎勵、擊敗敵人等
        
        custom_reward = get_coin_reward(info, reward, prev_info)
        custom_reward = distance_x_offset_reward(info, custom_reward, prev_info)
        custom_reward = distance_y_offset_reward(info, custom_reward, prev_info, action)
        custom_reward = monster_score_reward(info, custom_reward, prev_info)
        custom_reward = final_flag_reward(info, custom_reward)
        custom_reward = avoid_danger_reward(info, custom_reward, prev_info, action)
        custom_reward = consecutive_jump_reward(info, custom_reward, jump_streak)  

        # ===========================
        cumulative_custom_reward += custom_reward // 1

 

        # ===========================Check for x_pos stagnation  如果角色的水平位置未改變超過MAX_STAGNATION_STEPS則強制結束本局遊戲
        if info["x_pos"] == prev_info["x_pos"]:
            stagnation_time += 1
            if stagnation_time >= MAX_STAGNATION_STEPS:
                print(f"Timestep {timestep} - Early stop triggered due to x_pos stagnation.")
                done = True
        else:
            stagnation_time = 0
        
        
        #===========================Store transition in memory 將狀態轉移 (state, action, reward, next_state, done) 存入記憶體
        memory.push(state, action, (reward *2) + custom_reward //1, next_state, done)      #使用自訂義獎勵
        #memory.push(state, action, reward, next_state, done)                  #使用其預設好的獎勵
        #更新當前狀態
        state = next_state

        #==============================Train DQN 當記憶體中樣本數量達到批次大小時，從記憶體中隨機抽取一批樣本進行網路更新
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)

            state_dict = {                                       #將這些數據打包為字典格式，方便傳遞給模型進行訓練
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)                       #train_per_step是DQN中的方法，用於計算損失並更新神經網路的權重

        # Update epsilon
        dqn.epsilon = EPSILON_END               #訓練前就設定:代理的探索能力會立即降低，可能在策略還不完善時過早專注於利用，會影響最終的學習效果
        
        #================================更新狀態訊息
        prev_info = info
        step += 1

        if VISUALIZE:                                   #渲染當前遊戲畫面
            env.render()

    # Print cumulative reward for the current timestep
    print(f"Timestep {timestep} - Total Reward: {cumulative_reward} - Total Custom Reward: {cumulative_custom_reward}")


    # 定義保存資料夾，新增子資料夾
    save_folder = os.path.join("ckpt_test", "0108")
    official_save_folder = os.path.join("ckpt_test", "0108","official")
    last_timestep_folder = os.path.join("ckpt_test", "0108", "last_timesteps")
    os.makedirs(save_folder, exist_ok=True)  # 確保資料夾存在
    os.makedirs(official_save_folder, exist_ok=True)  # 確保官方資料夾存在
    #如果當前累積獎勵超過歷史最佳值，保存模型的權重 每次超過最佳值就會保留一次
    #要改成自定義獎勵
    # 儲存客製化最佳權重
    if cumulative_custom_reward > best_reward:
        best_reward = cumulative_custom_reward
        custom_model_path = os.path.join(
            save_folder,
            f"step_{timestep}_reward_{int(best_reward)}_custom_{int(cumulative_custom_reward)}.pth"
        )
        torch.save(dqn.q_net.state_dict(), custom_model_path)
        print(f"Custom reward model saved: {custom_model_path}")

    # 儲存官方最佳權重
    if cumulative_reward > official_best_reward:
        official_best_reward = cumulative_reward
        official_model_path = os.path.join(
            official_save_folder,
            f"step_{timestep}_reward_{int(official_best_reward)}_official.pth"
        )
        torch.save(dqn.q_net.state_dict(), official_model_path)
        print(f"Official reward model saved: {official_model_path}")
    # 儲存最後十個 Timestep 的權重
    if timestep > TOTAL_TIMESTEPS - 10:  # 檢查是否為最後十個 Timestep
        last_timestep_path = os.path.join(
            last_timestep_folder,
            f"step_{timestep}_reward_{int(cumulative_custom_reward)}.pth"
        )
        torch.save(dqn.q_net.state_dict(), last_timestep_path)
        print(f"Saved model for last timestep: {last_timestep_path}")

env.close()