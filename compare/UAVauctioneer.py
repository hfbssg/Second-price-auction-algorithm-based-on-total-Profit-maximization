import random

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from auction3 import Auction3
from secondUAVauctioneer import Auctioneer5
from firstUAVauctioneer import Auctioneer6

class Auctioneer4:

    def __init__(self, average_winner_bid, average_winner_fact_bid, winner, winner_profit, starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=6,
                        R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,
                        Fn=5e9, t=0.01, k_sellers=10e-6, Pn=3 ): #M_types=3, #, universal_maximum_price=100
        """
        :param penalty_factor: Multiplier for fee calculationz //费用计算乘数,用于计算损失的费用
        :param bidding_factor_strategy: Array with the bidding factor strategy of each buyer //每个买家的竞价因素策略
        :param use_seller: Flag to use seller or item as second dimension for alpha //标记使用卖家或项目作为alpha的第二个维度
        :param starting_prices: Debug purposes, starting prices can be forced this way. //出于调试目的，可以通过这种方式强制设定起始价格
        :param M_types: Number of types of items //商品类型的数量
        :param K_sellers: Number of sellers //卖家的数量
        :param N_buyers: Number of buyers //买家的数量
        :param R_rounds: Number of rounds //拍卖的轮数
        :param level_comm_flag: Flag to say if level commitment is allowed or not //承诺标记
        :param debug: Flag for debug prints //debug标记
        :param universal_maximum_price: Max initial starting price //最高初始价格
        """
        self.n_buyers = N_buyers #MD数量
        self.k_sellers = K_sellers #UAV数量
        self.r_rounds = R_rounds #拍卖轮数
        self.rE = rE #单位能耗的经济成本
        self.rC = rC #单位时延的经济成本
        self.k_m = k_buyers #MD能耗效率
        self.debug = debug #debug标记
        self.w = w #信道带宽
        self.n0 = n0 #噪声功率
        self.Gm = Gm #MD与UAV的信道增益
        self.delay = delay #排队时延
        self.Fn = Fn #UAV的计算能力
        self.t = t #每一轮拍卖的持续时间
        self.k_n= k_sellers #UAV能耗系数
        self.Pn = Pn #UAV的发送功率
        self.winner = winner
        self.winner_profit = winner_profit
        self.average_winner_bid = average_winner_bid  # 平均赢家投标
        self.average_winner_fact_bid = average_winner_fact_bid
        # if len(bidding_factor_strategy) == 0:
        #     # If the strategy is not passed, it is set to default 0
        #     # bidding_factor_strategy = [np.random.randint(2, 4, 1) for n in range(N_buyers)]
        #     bidding_factor_strategy = [2 for n in range(N_buyers)]
        # else:
        #     for s in bidding_factor_strategy:
        #         if s not in [1, 2, 3, 4]:
        #             print("Error in the strategy input")
        #             return
        # self.m_item_types = range(M_types)

        # self.max_starting_price = universal_maximum_price
        # self.penalty_factor = penalty_factor
        # If level commitment is activated sellers cannot cancel a won auction //如果水平承诺被激活，卖方不能取消赢家拍卖
        # self.level_commitment_activated = level_comm_flag
        self.auctions_history = []

        # # Assign a type of item to each seller randomly //给每个卖家随机分配一种商品
        # self.sellers_types = [random.sample(self.m_item_types, 1)[0] for seller in range(self.k_sellers)]

        # Assign second dimension of alpha following the input flag //依据输入标志设计alpha的第二个维度
        # if use_seller:
        #     self.second_dimension = self.k_sellers
        # else:
        #     self.second_dimension = M_types
        #
        # self.bidding_factor_strategy = bidding_factor_strategy

        self.bidding_factor = self.calculate_bidding_factor()
        self.increase_bidding_factor = np.random.uniform(1, 1.5, size=self.n_buyers)
        # self.decrease_bidding_factor = np.random.uniform(0.3, 0.8, size=self.n_buyers)

        # Ceiling threshold for strategy 2 //策略2的上限阈值
        # self.ceiling = 2

        # self.market_price = np.zeros((self.r_rounds, self.k_sellers))
        self.buyers_profits = np.zeros((self.r_rounds, self.n_buyers))
        self.cumulative_buyers_profits = np.zeros((self.n_buyers, self.r_rounds))
        self.cumulative_sellers_profits = np.zeros((self.k_sellers, self.r_rounds))
        self.cumulative_total_profits = np.zeros(self.k_sellers)
        self.sellers_total_profits = np.zeros(self.r_rounds)
        self.buyers_total_profits = np.zeros(self.r_rounds)
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))

        self.data = data
        self.cycles = cycles
        self.time = time
        self.f_m= f_m
        self.Pm = Pm
        self.starting_prices = self.calculate_starting_prices(starting_prices)
        # self.print_alphas()

        # self.times_items_returned = 0
        # self.times_bad_trade = 0

    def calculate_bid(self, buyer_id, starting_price):
        """
        Calculate the bid for a specific buyer considering his bidding strategy //计算出价为一个特定的买家考虑他的出价策略
        :param buyer_id: id of the buyer to calculate the bid from
        :param item_type: kind of item that is being auction
        :param seller_id: id of the seller that is auctioning
        :param starting_price: starting price of the item that is being auctioned
        :param auction_round: round of the auction
        :return: bid of the buyer
        """

        # second_dimension = seller_id
        # if self.second_dimension == len(self.m_item_types):
        #     second_dimension = item_type

        bid = self.bidding_factor[buyer_id] * starting_price
        # print(bid)

        # if not self.level_commitment_activated \
        #         or not self.buyers_already_won[buyer_id]:
        #     # If the buyer flag is not ON it means the buyer hasn't win an auction in this round yet #如果买方旗帜没有亮，这意味着买方还没有赢得这一轮拍卖
        #     return bid
        # auction, seller = self.get_auction_with_winner(buyer_id, auction_round)
        # previous_profit, market_price = auction.winner_profit, auction.market_price
        # penalty = self.calculate_fee(market_price - previous_profit)

        return bid

    def calculate_bidding_factor(self):
        """
        Bidding factor strategies: #计算出价因子1-2之间的随机数，一维
            1 - When an auction is won, the bidding factor is multiplied by the increasing factor and when lost by
                the decreasing factor
            2 - Depends on the kind of item, but has a max value to avoid price explosion.
                If alpha bigger than 2, decrease it using decrease factor.
            3 - Depends on the kind of item, if the bid is higher than market price, bidding factor is multiplied by
                the decreasing factor while if it is lower multiply by the increasing factor.
        """

        bidding_factor = []
        for seller in range(self.k_sellers):
            for buyer in range(self.n_buyers):
                bidding_factor.append(
                    np.random.uniform(5, 7)
                )
        # bidding_factor = np.array(bidding_factor)
        # print('bidding_factor=',bidding_factor)

        return bidding_factor

    def calculate_starting_prices(self, starting_prices):
        """
        Calculate the starting prices of the sellers. If the input parameter is empty they will be empty otherwise they
        will be the same as the input parameter, this is only for debug purposes.//计算卖家的起始要价价格
        :param starting_prices: DEBUG purposes. Set with the desired initial prices. If empty calculate them randomly.//如果为空则随机计算
        :return: the starting prices for the auctions
        """
        if len(starting_prices) > 0:
            return starting_prices
        starting_prices = []
        for seller in range(self.k_sellers):
                starting_prices.append(((self.rE * self.k_n * self.data) + ((self.rC * self.cycles) / self.Fn))) #random.random()生成0-1之间的随机浮点数
        # print('starting_prices=',starting_prices)
        return starting_prices

    # def calculate_fee(self, price_paid):
    #     # Calculate the fee to pay for an item if it is cancelled //如果商品被取消计算损失的费用
    #     return self.penalty_factor * price_paid

    # def choose_item_to_keep(self, auction, market_price, price_to_pay, winner, seller, auction_round):
    #     """
    #     When an buyers wins a second item in a round one of the items has to be returned. The agent is rational and
    #     therefore will always keep the item with higher return considering the fee to pay for the returned item.当买家在一轮中赢得第二件物品时，其中一件物品必须退还。智能体是理性的和
    #                                                                                                             因此，考虑到为归还物品支付的费用，将始终保留回报更高的物品。
    #     :param auction: auction object with the information of the auction that made the buyer win the new item
    #     :param market_price: market price of the item just won
    #     :param price_to_pay: price paid for the new item
    #     :param winner: id of the buyer
    #     :param seller: id of the seller
    #     :param auction_round: round of the auction
    #     """
    #
    #     self.times_items_returned += 1
    #     previous_auction, previous_seller = self.get_auction_with_winner(winner, auction_round)
    #     previous_winner_profit = previous_auction.winner_profit
    #     previous_fee = self.calculate_fee(previous_auction.price_paid)
    #     new_profit = market_price - price_to_pay
    #     new_fee = self.calculate_fee(price_to_pay)
    #
    #     if new_profit - previous_fee > previous_winner_profit - new_fee:
    #         # It is profitable to keep the new item, pay fee to previous seller
    #         previous_auction.return_item(previous_fee,
    #                                      kept_item_profit=new_profit,
    #                                      kept_item_fee=new_fee,
    #                                      seller_item_kept=seller,
    #                                      kept_item_price=price_to_pay)
    #
    #         if new_profit - previous_fee < 0:
    #             self.times_bad_trade += 1
    #     else:
    #         auction.return_item(new_fee,
    #                             kept_item_profit=previous_winner_profit,
    #                             kept_item_fee=previous_fee,
    #                             seller_item_kept=previous_seller,
    #                             kept_item_price=previous_auction.price_paid)
    #
    #         if previous_winner_profit - new_fee < 0:
    #             self.times_bad_trade += 1

    def choose_winner(self, bids, total_profit): #保证总利润最高
        """
        Chooose the winner of an auction. //选择拍卖的赢家 //选择投标最高的为赢家，并由支付第二规则决定支付 ???选择总利润最高的为赢家，第二支付规则???第一支付规则
        :param bids: map with the bids made by the buyers. Key is the id of the buyer and Value the bid 对应有买家的出价。key是买家的id，值是买家的出价
        :param market_price: market price of the item to sell //市场价格
        :return: id of the buyer that wins the item, price to pay by the winner //赢得商品的买家Id，由获胜者支付的价格
        """
        valid_bids = []
        fact_bids = []
        for bid in bids.values():

            # if bid > market_price:
            #     continue

            valid_bids.append(bid)

        fact_bids=sorted(valid_bids, reverse=True) #对买家投标进行降序排序

        total = []
        for profit in total_profit.values():

            # if bid > market_price:
            #     continue

            total.append(profit)

        total = sorted(total, reverse=True) #对总利润进行降序排序

        winner_id = [key1 for key1 in range(self.n_buyers) if total_profit[key1] == total[0]][0] #总利润最高为赢家

        if valid_bids[winner_id] == fact_bids[len(fact_bids)-1]:
            price_to_pay = valid_bids[winner_id]  # 赢家第二支付规则
        else:
            id = [key2 for key2 in range(self.n_buyers) if fact_bids[key2] == valid_bids[winner_id]][0]
            price_to_pay = fact_bids[id+1] # 赢家第二支付规则

        return winner_id, price_to_pay


    def start_auction(self, starting_prices):

        buyers_bid = {}
        total_profit = {}
        total = {}
        self.auctions_history.append([])
        starting_price = self.starting_prices[0]
        for seller in range(self.k_sellers):  # 10个卖家
            for buyer in range(self.n_buyers):  # 3个买家
                if seller == 0:
                    bid = self.calculate_bid(buyer, starting_price)
                    buyers_bid[buyer] = bid
                    auction1 = Auction3(starting_price=starting_price, buyer=buyer, price_paid=buyers_bid[buyer],
                                        bid_history=buyers_bid, data=self.data,
                                        cycles=self.cycles, f_m=self.f_m, Pm=self.Pm)
                    buyer_bid = auction1.buyer_profit[0]  # 计算买家利润
                    seller_bid = auction1.seller_profit[0]  # 计算卖家利润
                    total_profit[buyer] = seller_bid + buyer_bid
                else:
                    bid = self.calculate_bid(buyer, starting_price)
                    buyers_bid[buyer] = bid
                    auction3 = Auction3(starting_price=starting_price, buyer=buyer, price_paid=buyers_bid[buyer],
                                                bid_history=buyers_bid, data=self.data,
                                                cycles=self.cycles, f_m=self.f_m, Pm=self.Pm)
                    seller_bid = auction3.seller_profit[0]  # 根据投标计算卖家利润 不是实际利润
                    buyer_bid = auction3.buyer_profit[0]  # 根据投标计算买家利润

                    total_bid_profit = seller_bid + buyer_bid
                    total_profit[buyer] = total_bid_profit  # 根据投标计算每个MD的总利润

            # market_price = total_bid / n_buyer_auction
            winner, price_to_pay = self.choose_winner(buyers_bid, total_profit)
            auction2 = Auction3(starting_price=starting_price, buyer=buyer, price_paid=price_to_pay,
                                            bid_history=buyers_bid, data=self.data,
                                            cycles=self.cycles, f_m=self.f_m, Pm=self.Pm)
            winner_bid = auction2.buyer_profit[0]  # 计算实际赢家利润
            seller_bid = auction2.seller_profit[0]  # 计算实际卖家利润
            total[seller] = seller_bid + winner_bid

            if seller == 0:
                self.cumulative_total_profits[seller] = total[seller]

            elif seller == 1:
                self.cumulative_total_profits[seller] = total[seller] + total[seller-1]  # 根据投标计算整个拍卖的实际总利润

            elif seller == 2:
                self.cumulative_total_profits[seller] = total[seller] + total[seller-1] + total[seller-2]

            elif seller == 3:
                self.cumulative_total_profits[seller] = total[seller] + total[seller-1] + total[seller-2] + total[seller-3]

            elif seller == 4:
                self.cumulative_total_profits[seller] = total[seller] + total[seller-1] + total[seller-2] + total[seller-3]+ total[seller-4]

            elif seller == 5:
                self.cumulative_total_profits[seller] = total[seller] + total[seller-1] + total[seller-2] + total[seller-3] + total[seller-4] + total[seller-5]

            elif seller == 6:
                self.cumulative_total_profits[seller] = total[seller] + total[seller - 1] + total[seller - 2] + total[
                    seller - 3] + total[seller - 4] + total[seller - 5] + total[seller - 6]

            elif seller == 7:
                self.cumulative_total_profits[seller] = total[seller] + total[seller - 1] + total[seller - 2] + total[
                    seller - 3] + total[seller - 4] + total[seller - 5] + total[seller - 6] + total[seller - 7]

            elif seller == 8:
                self.cumulative_total_profits[seller] = total[seller] + total[seller - 1] + total[seller - 2] + total[
                    seller - 3] + total[seller - 4] + total[seller - 5] + total[seller - 6] + total[seller - 7] + total[seller - 8]

            elif seller == 9:
                self.cumulative_total_profits[seller] = total[seller] + total[seller - 1] + total[seller - 2] + total[
                    seller - 3] + total[seller - 4] + total[seller - 5] + total[seller - 6] + total[seller - 7] + total[seller - 8] + total[seller - 9]

            else:
                self.cumulative_total_profits[seller] = total[seller] + total[seller - 1] + total[seller - 2] + total[
                    seller - 3] + total[seller - 4] + total[seller - 5] + total[seller - 6] + total[seller - 7] + total[seller - 8] + total[seller - 9] + total[seller - 10]




    def store_auction_history(self, starting_price, buyer, price_paid, bid_history, auction_round, data, cycles, f_m, Pm):
        """
        Store the information of an auction in an auction object and store it in the auctions history //在拍卖对象中存储拍卖信息，并将其存储在拍卖历史记录中
        :param starting_price: Starting price of the auction
        :param market_price: market price of the item
        :param winner: id of the buyer that wins the auction
        :param price_paid: price that the buyer pays for the item
        :param bid_history: dictionary with the bid of the buyers
        :param previous_alphas: bidding factor before the auction
        :param auction_round: round that this auction took place in
        :param item_kind: kind of item that is sold
        :return: auction object with all the information //返回拍卖的所有信息
        """
        auction = Auction3(starting_price, buyer, price_paid, bid_history, auction_round, data, cycles, f_m, Pm)
        self.auctions_history[auction_round].append(auction) #记录拍卖历史
        return auction

    def plot_statistics(self):
        """
        Plot the statistics of the history of the prices, the profit of the buyers and the sellers #画出历史价格的统计数据，买家和卖家的利润
\       """

        # Plot total profits //每一轮总利润三种方式的比较,MD的个数
        # plt.figure()
        # x = []
        # for i in range(self.k_sellers):
        #     x.append(i + 1)
        #
        # a_auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], cumulative_total_profits=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # a_auctioneer.start_auction(starting_prices=[])
        # a_auctioneer.cumulative_total_profits
        #
        # b_auctioneer = Auctioneer2(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # b_auctioneer.start_auction(starting_prices=[])
        # b_auctioneer.cumulative_total_profits
        #
        # c_auctioneer = Auctioneer3(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], cumulative_total_profits=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=10, K_sellers=3,
        #                     R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # c_auctioneer.start_auction(starting_prices=[])
        # c_auctioneer.cumulative_total_profits
        #
        # y_1 = a_auctioneer.cumulative_total_profits
        # y_2 = b_auctioneer.cumulative_total_profits # y轴的值
        # y_3 = c_auctioneer.cumulative_total_profits
        #
        # plt.grid(True, linestyle='--', alpha=0.5)
        # plt.plot(x, y_1, color='orangered', linestyle='-', label='second-price')
        # plt.plot(x, y_2, color='blueviolet', linestyle='-.', label='first-price')
        # plt.plot(x, y_3, color='green', linestyle=':', label='proposed')
        # plt.legend()  # 显示图例
        # plt.title('total cumulative profits across UAVs numbers')
        # plt.ylabel('total profits')
        # plt.xlabel('UAVs numbers')
        # plt.show()

        # Plot total profits //每一轮总利润三种方式的比较,UAV的个数
        plt.figure()
        x = []
        for i in range(self.k_sellers):
            x.append(i + 1)

        # a_auctioneer = Auctioneer5(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], cumulative_total_profits=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=10,
        #                     R_rounds=20, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
        #                     k_sellers=10e-6, Pn=3)
        # a_auctioneer.start_auction(starting_prices=[])
        # a_auctioneer.cumulative_total_profits
        #
        # b_auctioneer = Auctioneer6(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[],
        #                            starting_prices=[], cumulative_total_profits=[], data=5e5, cycles=10e8, time=0.8,
        #                            f_m=1e9,
        #                            Pm=0.3, N_buyers=3, K_sellers=10, R_rounds=20, rE=0.1, rC=60, k_buyers=10e-27,
        #                            debug=True, w=10e6, Gm=300, n0=10, delay=0.02,
        #                            Fn=5e9, t=0.01, k_sellers=10e-6, Pn=3)
        # b_auctioneer.start_auction(starting_prices=[])
        # b_auctioneer.cumulative_total_profits

        # y_1 = a_auctioneer.cumulative_total_profits
        # y_2 = b_auctioneer.cumulative_total_profits # y轴的值
        y_3 = self.cumulative_total_profits

        plt.grid(True, linestyle='--', alpha=0.5)
        # plt.plot(x, y_1, color='orangered', linestyle='-', label='second-price')
        # plt.plot(x, y_2, color='blueviolet', linestyle='-.', label='first-price')
        plt.plot(x, y_3, color='green', linestyle=':', label='proposed')
        plt.legend()  # 显示图例
        plt.title('total cumulative profits across UAVs numbers')
        plt.ylabel('total profits')
        plt.xlabel('UAVs numbers')
        plt.show()

if __name__ == '__main__':
    # strategy = [1 for n in range(buyers)]
    # strategy[0] = 4
    auctioneer = Auctioneer4(average_winner_bid=[], average_winner_fact_bid=[], winner=[], winner_profit=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=10,
                            R_rounds=20, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.01,
                            k_sellers=10e-6, Pn=3)
    auctioneer.start_auction(starting_prices=[])
    auctioneer.plot_statistics()
    print("\n the simulation is finished")
    # auctioneer.print_alphas()
