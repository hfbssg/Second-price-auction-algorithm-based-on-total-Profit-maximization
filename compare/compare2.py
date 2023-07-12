import random

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# from auction import Auction
# from auctioneer import Auctioneer1
# from firstauction import Auctioneer2
# from profitauction import Auction1
# from profit import Auctioneer3

from MDauctioneer import Auctioneer3
from secondMDauctioneer import Auctioneer1
from firstMDauctioneer import Auctioneer2

from UAVauctioneer import Auctioneer4
from secondUAVauctioneer import Auctioneer5
from firstUAVauctioneer import Auctioneer6



class Auctioneer:

    def __init__(self, N_buyers, K_sellers):

        self.n_buyers = N_buyers #拍卖中MD个数
        self.k_sellers = K_sellers #拍卖中UAV个数

    def plot_statistics(self):
        """
        Plot the statistics of the history of the prices, the profit of the buyers and the sellers #画出历史价格的统计数据，买家和卖家的利润
\       """

        # Plot total profits //每一轮总利润三种方式的比较,MD的个数
        plt.figure()
        x = []
        for i in range(self.n_buyers):
            x.append(i + 1)

        a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], cumulative_total_profits=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
                            k_sellers=10e-6, Pn=3)
        a_auctioneer.start_auction(starting_prices=[])
        a_auctioneer.cumulative_total_profits

        b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
                            k_sellers=10e-6, Pn=3)
        b_auctioneer.start_auction(starting_prices=[])
        b_auctioneer.cumulative_total_profits

        c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], cumulative_total_profits=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
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
        plt.xlabel('MDs的数量')
        plt.show()

        # Plot total profits //每一轮总利润三种方式的比较,UAV的个数
        # plt.figure()
        # x = []
        # for i in range(self.k_sellers):
        #     x.append(i + 1)
        #
        # a_auctioneer = Auctioneer5(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=10,
        #                     R_rounds=20, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # a_auctioneer.start_auction(starting_prices=[])
        # a_auctioneer.cumulative_total_profits
        #
        # b_auctioneer = Auctioneer6(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9,
        #                      Pm=0.3, N_buyers=3, K_sellers=10, R_rounds=20, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,
        #                      Fn=5e9, t=0.01,k_sellers=10e-6, Pn=3)
        # b_auctioneer.start_auction(starting_prices=[])
        # b_auctioneer.cumulative_total_profits
        #
        # c_auctioneer = Auctioneer4(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=10,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # c_auctioneer.start_auction(starting_prices=[])
        # c_auctioneer.cumulative_total_profits
        #
        # y_1 = a_auctioneer.cumulative_total_profits
        # y_2 = b_auctioneer.cumulative_total_profits # y轴的值
        # y_3 = c_auctioneer.cumulative_total_profits
        #
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['font.size'] = 12
        # plt.grid(True, linestyle='--', alpha=1)
        # plt.plot(x, y_1, color='orangered',linewidth=2, linestyle='-', label='第二价格拍卖算法', marker = "o", markersize=6)
        # plt.plot(x, y_2, color='blueviolet', linewidth=2, linestyle='-.', label='第一价格拍卖算法', marker = "x", markersize=6)
        # plt.plot(x, y_3, color='green', linewidth=2, linestyle=':', label='所提算法', marker = "^", markersize=6)
        # plt.legend()  # 显示图例
        # plt.ylabel('积累总利润')
        # plt.xlabel('UAVs的数量')
        # plt.show()




if __name__ == '__main__':

    auctioneer = Auctioneer(N_buyers=10, K_sellers=10)
    auctioneer.plot_statistics()
    print("\n the simulation is finished")
