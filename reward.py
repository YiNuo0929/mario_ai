import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state 
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# # simple actions_dim = 7 
# SIMPLE_MOVEMENT = [
#     ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], #M ove right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
# ]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
def get_coin_reward(info, reward, prev_info):
    """
    獎勵蒐集硬幣的行為。
    """
    total_reward = reward
    coin_reward = (info['coins'] - prev_info['coins']) * 3  # 每枚硬幣給予 3 分
    total_reward += coin_reward
    return total_reward


def distance_y_offset_reward(info, reward, prev_info, action):
    """
    獎勵跳躍行為，根據 y_pos 的變化給予獎勵，鼓勵有意義的跳躍行為。
    """
    total_reward = reward
    y_change = info['y_pos'] - prev_info['y_pos']  # 計算垂直方向變化

    if y_change > 0:
        if action in [2, 4]:  # 鼓勵右跳或右跑跳
            jump_reward = y_change * 1  # 每單元跳躍高度給 1 分
        else:  # 無意義的垂直跳躍
            jump_reward = 0.1  # 只加少量獎勵

        total_reward += jump_reward

    return total_reward


def distance_x_offset_reward(info, reward, prev_info):
    """
    獎勵前進行為，懲罰停滯或後退行為。
    """
    total_reward = reward
    x_change = info['x_pos'] - prev_info['x_pos']  # 計算水平方向變化

    if x_change > 0:
        x_reward = x_change * 3  # 每單元前進給 3 分
        total_reward += x_reward
    elif x_change == 0:
        penalty = -20  # 停滯扣 20 分
        total_reward += penalty
    elif x_change < 0:
        back_penalty = x_change * 10  # 每單元後退扣 10 分
        total_reward += back_penalty
    return total_reward


def monster_score_reward(info, reward, prev_info):
    """
    獎勵擊敗敵人或增加遊戲分數的行為。
    """
    total_reward = reward
    score_change = info['score'] - prev_info['score']  # 計算分數變化

    if score_change > 0:
        total_reward += score_change * 200  # 每增加 1 分，給予 200 分獎勵
    return total_reward


def final_flag_reward(info, reward):
    """
    獎勵到達終點旗幟的行為。
    """
    total_reward = reward
    if info['flag_get']:
        total_reward += 1000  # 完成關卡給 1000 分
    return total_reward


def avoid_danger_reward(info, reward, prev_info, action):
    """
    鼓勵避開怪物和坑洞。
    """
    total_reward = reward

    # 檢查生命數是否減少
    if info["life"] < prev_info["life"]:
        penalty = -5000  # 碰到怪物或掉進洞穴給予懲罰
        total_reward += penalty

    # 獎勵持續存活
    else:
        survival_reward = 0.2  # 每存活一個回合給予獎勵
        total_reward += survival_reward

    return total_reward

def consecutive_jump_reward(info, reward, jump_streak):
    """
    獎勵 Mario 的連續跳躍行為。
    """
    if jump_streak > 10:  # 如果連續跳躍次數超過 10
        streak_bonus = 3 + (jump_streak - 10) * 0.5  # 每次額外增加 0.5 分
        reward += streak_bonus
        #print(f"連續跳躍次數: {jump_streak}, 獎勵: {streak_bonus}")
    return reward

'''
def time_remaining_reward(info, reward):
    """
    獎勵在有限時間內快速完成。
    """
    total_reward = reward
    time_bonus = info['time'] * 2  # 根據剩餘時間給予獎勵，每秒 2 分
    total_reward += time_bonus
    return total_reward
'''

#===============to do==========================================