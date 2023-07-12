import random

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from auction import Auction
from auctioneer import Auctioneer1
from firstauction import Auctioneer2
from profitauction import Auction1
from profit import Auctioneer3

class Auctioneer:

    def __init__(self, R_rounds):

        self.r_rounds = R_rounds #拍卖轮数


    def plot_statistics1(self):
        """
        Plot the statistics of the history of the prices, the profit of the buyers and the sellers #画出历史价格的统计数据，买家和卖家的利润
\       """
        # #  Plot total profits //每一轮总卖家利润三种方式的比较,轮次
        # plt.figure()
        # x = range(self.r_rounds)
        #
        # a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
        #                     k_sellers=10e-6, Pn=3)
        # a_auctioneer.start_auction(starting_prices=[])
        # # a_auctioneer.cumulative_total_profits
        # a_auctioneer.sellers_total_profits
        #
        #
        # b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
        #                     k_sellers=10e-6, Pn=3)
        # b_auctioneer.start_auction(starting_prices=[])
        # # b_auctioneer.cumulative_total_profits
        # b_auctioneer.sellers_total_profits
        #
        # c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
        #                     k_sellers=10e-6, Pn=3)
        # c_auctioneer.start_auction(starting_prices=[])
        # # c_auctioneer.cumulative_total_profits
        # c_auctioneer.sellers_total_profits
        #
        # # y_1 = a_auctioneer.cumulative_total_profits
        # # y_2 = b_auctioneer.cumulative_total_profits # y轴的值
        # # y_3 = c_auctioneer.cumulative_total_profits
        #
        # y_1 = a_auctioneer.sellers_total_profits
        # y_2 = b_auctioneer.sellers_total_profits # y轴的值
        # y_3 = c_auctioneer.sellers_total_profits
        #
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.plot(x, y_1, color='orangered',  linestyle='-', label='second-price')
        # plt.plot(x, y_2, color='blueviolet',  linestyle='-.', label='first-price')
        # plt.plot(x, y_3, color='green',  linestyle=':', label='proposed')
        # plt.legend()  # 显示图例
        # plt.title('total sellers cumulative profits across all auctions')
        # plt.ylabel('total sellers profits')
        # plt.xlabel('Rounds')


        #  Plot total profits //每一轮总利润三种方式的比较,轮次
        plt.figure()
        x = []
        for i in range(self.r_rounds):
            x.append(i + 1)

        a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
                            k_sellers=10e-6, Pn=3)
        a_auctioneer.start_auction(starting_prices=[])
        a_auctioneer.cumulative_total_profits

        b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
                            k_sellers=10e-6, Pn=3)
        b_auctioneer.start_auction(starting_prices=[])
        b_auctioneer.cumulative_total_profits

        c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
                            k_sellers=10e-6, Pn=3)
        c_auctioneer.start_auction(starting_prices=[])
        c_auctioneer.cumulative_total_profits


        y_1 = a_auctioneer.cumulative_total_profits
        y_2 = b_auctioneer.cumulative_total_profits # y轴的值
        y_3 = c_auctioneer.cumulative_total_profits


        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['font.size'] = 12
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.plot(x, y_1, color='orangered',linewidth=2, linestyle='-', label='第二价格拍卖算法', marker = "o", markersize=6)
        plt.plot(x, y_2, color='blueviolet', linewidth=2, linestyle='-.', label='第一价格拍卖算法', marker = "x", markersize=6)
        plt.plot(x, y_3, color='green', linewidth=2, linestyle=':', label='所提算法', marker = "^", markersize=6)
        plt.legend()  # 显示图例
        plt.ylabel('积累总利润')
        plt.xlabel('轮数')
        plt.show()


        # Plot total profits //三种方法条形图的比较
    # def plot_statistics2(self):
    #     # 赢家的平均初始投标
    #
    #     x1 = []
    #     for i in range(self.r_rounds):
    #         x1.append(i + 1)
    #     x2 = list(map(lambda x: x + 0.2, x1))
    #     x3 = list(map(lambda x: x + 0.4, x1))
    #
    #     a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[],data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
    #                         R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
    #                         k_sellers=10e-6, Pn=3)
    #     a_auctioneer.start_auction(starting_prices=[])
    #
    #
    #     b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[],data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
    #                         R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
    #                         k_sellers=10e-6, Pn=3)
    #     b_auctioneer.start_auction(starting_prices=[])
    #
    #
    #     c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
    #                         R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
    #                         k_sellers=10e-6, Pn=3)
    #     c_auctioneer.start_auction(starting_prices=[])
    #
    #     plt.rcParams['font.sans-serif'] = ['SimHei']
    #     plt.rcParams['font.size'] = 12
    #
    #     plt.figure()
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.bar(x1, a_auctioneer.average_winner_bid, width=0.3, color='r', label='第二价格拍卖算法')
    #     plt.bar(x2, b_auctioneer.average_winner_bid, width=0.3, color='g', label='第一价格拍卖算法')
    #     plt.bar(x3, c_auctioneer.average_winner_bid, width=0.3, color='b', label='所提算法')
    #     plt.xticks(x2, x1)
    #     plt.ylim(5, 150)  # 设置y轴的显示范围
    #     plt.legend()
    #     plt.ylabel('赢家平均投标')
    #     plt.xlabel('轮数')
    #
    #     plt.figure()
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.bar(x1, a_auctioneer.average_winner_fact_bid, width=0.3, color='r', label='第二价格拍卖算法')
    #     plt.bar(x2, b_auctioneer.average_winner_fact_bid, width=0.3, color='g', label='第一价格拍卖算法')
    #     plt.bar(x3, c_auctioneer.average_winner_fact_bid, width=0.3, color='b', label='所提算法')
    #     plt.xticks(x2, x1)
    #     plt.ylim(5, 150)  # 设置y轴的显示范围
    #     plt.legend()
    #     plt.ylabel('赢家的平均实际支付')
    #     plt.xlabel('轮数')
    #     plt.show()

        # 赢家的平均实际支付
        # plt.figure()
        # x1 = []
        # for i in range(self.r_rounds):
        #     x1.append(i + 1)
        # x2 = list(map(lambda x: x + 0.2, x1))
        # x3 = list(map(lambda x: x + 0.4, x1))
        #
        # a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8,
        #                            time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                            R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10,
        #                            delay=0.02, Fn=5e9, t=0.001,
        #                            k_sellers=10e-6, Pn=3)
        # a_auctioneer.start_auction(starting_prices=[])
        #
        # b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8,
        #                            time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                            R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10,
        #                            delay=0.02, Fn=5e9, t=0.001,
        #                            k_sellers=10e-6, Pn=3)
        # b_auctioneer.start_auction(starting_prices=[])
        #
        # c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], starting_prices=[], data=5e5,
        #                            cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
        #                            R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10,
        #                            delay=0.02, Fn=5e9, t=0.001,
        #                            k_sellers=10e-6, Pn=3)
        # c_auctioneer.start_auction(starting_prices=[])
        #
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.bar(x1, a_auctioneer.average_winner_fact_bid, width=0.3, color='r', label='第二价格拍卖算法')
        # plt.bar(x2, b_auctioneer.average_winner_fact_bid, width=0.3, color='g', label='第一价格拍卖算法')
        # plt.bar(x3, c_auctioneer.average_winner_fact_bid, width=0.3, color='b', label='所提算法')
        # plt.xticks(x2, x1)
        # plt.ylim(5, 150)  # 设置y轴的显示范围
        # plt.legend()
        # plt.title('赢家平均实际支付图')
        # plt.ylabel('赢家的平均实际支付')
        # plt.xlabel('轮数')
        # plt.show()




if __name__ == '__main__':

    auctioneer = Auctioneer(R_rounds=10)
    auctioneer.plot_statistics1()
    # auctioneer.plot_statistics2()
    print("\n the simulation is finished")
