import random


import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from auction import Auction


class Auctioneer1:

    def __init__(self, winner, average_winner_bid, average_winner_fact_bid, starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
                        R_rounds=6, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,
                        Fn=5e9, t=0.001, k_sellers=10e-6, Pn=3 ): #M_types=3, #, universal_maximum_price=100
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
        self.average_winner_bid = average_winner_bid #平均赢家投标
        self.average_winner_fact_bid = average_winner_fact_bid #平均赢家实际支付
        self.winner = winner #赢家

        self.data = data
        self.cycles = cycles
        self.time = time
        self.f_m = f_m
        self.Pm = Pm

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
        self.sellers_total_profits = np.zeros(self.r_rounds)
        self.buyers_total_profits = np.zeros(self.r_rounds)
        self.cumulative_total_profits = np.zeros(self.r_rounds)
        self.sellers_profits = np.zeros((self.r_rounds, self.k_sellers))
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
                    np.random.uniform(1, 1.2)
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
            starting_prices.append(((self.rE * self.k_n * self.data) + ((self.rC * self.cycles) / self.Fn)))
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

    def choose_winner(self, bids):
        """
        Chooose the winner of an auction. //选择拍卖的赢家 //选择投标最高的为赢家，并由支付第二规则决定支付 ???选择总利润最高的为赢家，第二支付规则???第一支付规则
        :param bids: map with the bids made by the buyers. Key is the id of the buyer and Value the bid 对应有买家的出价。key是买家的id，值是买家的出价
        :param market_price: market price of the item to sell //市场价格
        :return: id of the buyer that wins the item, price to pay by the winner //赢得商品的买家Id，由获胜者支付的价格
        """
        valid_bids = []
        for bid in bids.values():

            # if bid > market_price:
            #     continue

            valid_bids.append(bid)

        if len(valid_bids) == 0:
            valid_bids.append(next(iter(bids.values())))

        valid_bids = sorted(valid_bids, reverse=True) #降序排序

        winner_id = [key for key in bids.keys() if bids[key] == valid_bids[0]][0]

        price_to_pay = valid_bids[1] #支付第二出价 price_to_pay = valid_bids[0] #支付第一出价

        return winner_id, price_to_pay

    def get_auction_with_winner(self, winner, auction_round):
        """
        Retrieve the auction object of a previous auction with the winner. Used when level commitment is activated and
        a buyer wins a second time. #检索具有获胜者的先前拍卖的拍卖对象。当级别承诺被激活并且买家赢得第二次时使用。
        :param winner: id of the winner
        :param auction_round: round of the auction
        :return: auction object, seller id of the auction
        """
        seller = 0
        for auction in self.auctions_history[auction_round]:
            if winner == auction.winner:
                return auction, seller
            seller += 1
        assert 0 == 1

    def initialize_auction_parameters(self, seller):
        # Initialize all the parameters needed for an auction #初始化拍卖所需的所有参数
        starting_price = self.starting_prices[seller]

        # item = self.sellers_types[seller]
        # item = 0
        return starting_price

    def initialize_buyers_flag(self):
        # Initialize the list with the flags that indicates if a buyer has already won an auction in the round //用标记初始化列表，这些标记表示是否有买家已经在这一轮拍卖中获胜
        return [False for buyer in range(self.n_buyers)]

    # def print_alphas(self, extra_debug=False):
    #     """
    #     Print the values of the bidding factors. //打印投标因素的值
    #     :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
    #     required. With this input parameter this is possible.
    #     """
    #     if not self.debug and not extra_debug:
    #         return
    #
    #     buyer = 0
    #     alphas_table = PrettyTable()
    #
    #     if self.second_dimension == self.k_sellers:
    #         alphas_table.field_names = ["S-0"] + ["S" + str(seller) for seller in range(self.k_sellers)]
    #     elif self.second_dimension == len(self.m_item_types):
    #         alphas_table.field_names = ["S-1"] + ["Type " + str(item_type) for item_type in self.m_item_types]
    #
    #     for strategy in self.bidding_factor_strategy:
    #         alphas_table.add_row(["B" + str(buyer)] + ['%.2f' % elem for elem in self.bidding_factor[buyer]])
    #         str_0 = True
    #         buyer += 1
    #
    #     print(alphas_table)

    def print_factors(self, extra_debug=False):
        """
        Print the increasing and decreasing factors for every buyer. //打印每个买家的增减因子。
        :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
        required. With this input parameter this is possible.
        """
        if not self.debug and not extra_debug:
            return
        initial_table = PrettyTable()
        initial_table.field_names = [""] + ["B" + str(buyer) for buyer in range(self.n_buyers)]
        initial_table.add_row(["Increasing factor"] + ['%.2f' % elem for elem in self.increase_bidding_factor])
        # initial_table.add_row(["Decreasing factor"] + ['%.2f' % elem for elem in self.decrease_bidding_factor])
        print(initial_table)

#     def print_round(self, round_number, extra_debug=False):
#         """
#         Print the information of all the auctions in a round 在一个回合内打印所有拍卖的信息
#         :param round_number: round of auction
#         :param extra_debug: Even if in the parent object debug is set to false, it is possible that this printing is
#         required. With this input parameter this is possible. 即使在父对象中调试被设置为false，也有可能打印为必需的。有了这个输入参数，这是可能的
# \        """
#         if not self.debug and not extra_debug:
#             return
#         print()
#         print("Round", round_number, "history")
#         seller = 0
#         for auction in self.auctions_history[round_number]:
#             auction.print_auction(seller)
#             seller += 1
#         print()
#         print("------------------------------------------------------")

    # def update_alphas(self, winner, seller, item, bids):
    #     """
    #     Update the bidding factor depending on the strategies of each buyer //根据每个买家的策略更新出价因素,更新出价的影响因素
    #     :param winner: id of the winner of the auction
    #     :param seller: seller of the item of the auction
    #     :param item: kind of items that the seller auctions
    #     :param bids: dictionary with the bids of the buyers, key is the id of the buyer and the value is the bid
    #     :return: new alphas after updating //返回更新的出价因素
    #     """
    #
    #     second_dimension = seller
    #     if self.second_dimension == len(self.m_item_types):
    #         second_dimension = item
    #
    #     new_alphas = []
    #     for buyer in range(self.n_buyers):
    #         if self.bidding_factor_strategy[buyer] == 1:
    #             if buyer == winner:
    #                 self.bidding_factor[buyer][second_dimension] *= self.decrease_bidding_factor[buyer]
    #
    #             elif self.buyers_already_won[buyer] and not self.level_commitment_activated:
    #                 self.bidding_factor[buyer][second_dimension] = self.bidding_factor[buyer][second_dimension]
    #
    #             else:
    #                 self.bidding_factor[buyer][second_dimension] *= self.increase_bidding_factor[buyer]
    #
    #             new_alphas.append(self.bidding_factor[buyer][second_dimension])
    #
    #     return new_alphas

    def update_profits(self, auction_round):  # 实际利润
        """
        Update the profit of every buyer and seller after a round is finished 更新每个买家和卖家的利润
        :param auction_round: number of round
        """
        seller = 0
        for auction in self.auctions_history[auction_round]:
            self.buyers_profits[auction_round, self.winner[seller+self.k_sellers*auction_round]] += auction.winner_profit[0] #计算本轮拍卖的赢家利润
            self.sellers_profits[auction_round, seller] += auction.seller_profit[0] #计算本轮拍卖的卖家利润
            seller += 1

        for buyer in range(self.n_buyers):
            self.cumulative_buyers_profits[buyer][auction_round] = self.cumulative_buyers_profits[
                                                                       buyer, auction_round - 1] + self.buyers_profits[
                                                                       auction_round, buyer]
            self.buyers_total_profits[auction_round] += self.cumulative_buyers_profits[buyer][auction_round]

        for seller in range(self.k_sellers):
            self.cumulative_sellers_profits[seller][auction_round] = self.cumulative_sellers_profits[
                                                                         seller, auction_round - 1] + \
                                                                     self.sellers_profits[auction_round, seller]

            self.sellers_total_profits[auction_round] += self.cumulative_sellers_profits[seller][auction_round]

        self.cumulative_total_profits[auction_round] = self.buyers_total_profits[auction_round] + \
                                                       self.sellers_total_profits[auction_round]

    def start_auction(self, starting_prices):
        """
        Main method of the program, runs the actual simulation //程序的主要方法，运行实际仿真
        """
        self.print_factors() #打印拍卖的增减因子
        buyers_bid = {}
        for auction_round in range(self.r_rounds): #50轮
            winner_total = 0
            winner_fact_total = 0
            self.auctions_history.append([])

            for seller in range(self.k_sellers): #3个卖家
                starting_price = self.initialize_auction_parameters(seller)
                for buyer in range(self.n_buyers): #3个买家
                    if auction_round != 0 and buyer != self.winner[seller + self.k_sellers * (auction_round - 1)]: #确保是同一个卖家的赢家
                        buyers_bid[buyer] = buyers_bid[buyer] * np.random.uniform(1.1, 1.4) #败者提高投标增加竞争力
                    elif auction_round != 0 and buyer == self.winner[seller + self.k_sellers * (auction_round - 1)]:
                        buyers_bid[winner] = winner_bid #第二价格拍卖，下一轮赢家保持投标不变
                    else:
                        bid = self.calculate_bid(seller * self.n_buyers + buyer, starting_price)
                        buyers_bid[buyer] = bid

                # market_price = total_bid / n_buyer_auction
                winner, price_to_pay = self.choose_winner(buyers_bid)
                winner_bid = buyers_bid[winner]
                winner_total += winner_bid
                winner_fact_bid = price_to_pay
                winner_fact_total += winner_fact_bid
                auction = self.store_auction_history(winner=winner,
                                                     price_paid=price_to_pay,
                                                     starting_price=starting_price,
                                                     bid_history=buyers_bid,
                                                     auction_round=auction_round,
                                                     data=self.data,
                                                     cycles=self.cycles,
                                                     f_m=self.f_m,
                                                     Pm=self.Pm)
                self.winner.append(winner)

                # auction = self.store_auction_history(self.data, self.cycles, winner=winner,
                #                                      price_paid=price_to_pay,
                #                                      starting_price=starting_price,
                #                                      market_price=market_price,
                #                                      bid_history=buyers_bid,
                #                                      previous_alphas=self.get_alphas(seller, item),
                #                                      auction_round=auction_round,
                #                                      item_kind=item,
                #                                      rE=60, rC=50, Pn=3, Fn=5e9, k_sellers=10e-4)

                # if self.level_commitment_activated and self.buyers_already_won[winner]:
                #     # The buyer already won an auction in this round so he has to choose which one to return //买家已经在这一轮拍卖中赢得了一场胜利，所以他必须选择退回哪一场
                #     self.choose_item_to_keep(auction, market_price, price_to_pay, winner, seller, auction_round)

                # self.market_price[auction_round, seller] = market_price
                # new_alphas = self.update_alphas(winner, seller, item, buyers_bid)
                # auction.set_new_alphas(new_alphas)

            self.update_profits(auction_round)
            # self.print_round(auction_round)
            self.average_winner_bid.append(winner_total / self.k_sellers)
            self.average_winner_fact_bid.append(winner_fact_total / self.k_sellers)

    def store_auction_history(self, starting_price, winner, price_paid, bid_history, auction_round, data, cycles, f_m, Pm):
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
        auction = Auction(starting_price, winner, price_paid, bid_history, auction_round, data, cycles, f_m, Pm)
        self.auctions_history[auction_round].append(auction)
        return auction

    def plot_statistics(self):
        """
        Plot the statistics of the history of the prices, the profit of the buyers and the sellers #画出历史价格的统计数据，买家和卖家的利润
\       """
        # market_prices = np.zeros((self.r_rounds, self.k_sellers))

        # for n, auctions_round in enumerate(self.auctions_history): #enumerate得到一个索引序列
        #     for seller in range(self.k_sellers):
        #         market_prices[n, seller] = auctions_round[seller].market_price

        # # Plot price history //历史价格图
        # for seller in range(self.k_sellers):
        #     if self.bidding_factor_strategy[0] == 1:
        #         plt.semilogy(market_prices[:, seller], label="Seller " + str(seller)) #matplotlib.pyplot.semilogy()函数用于绘制在y轴上具有对数刻度的图。
        #     else:
        #         plt.plot(market_prices[:, seller], label="Seller " + str(seller))
        #
        # plt.title('Price history across all rounds for each seller')
        # plt.ylabel('Price')
        # plt.xlabel('Auctions')
        # plt.legend() #设置图例

        # Plot seller profits //卖家利润图
        plt.figure()
        plt.grid(True, linestyle='--', alpha=0.5)
        for seller in range(self.k_sellers):
            plt.semilogy(self.cumulative_sellers_profits[seller], label="Seller " + str(seller)) #matplotlib.pyplot.semilogy()函数用于绘制在y轴上具有对数刻度的图。

        plt.title('Seller cumulative profits across all auctions')
        plt.ylabel('Seller profits')
        plt.xlabel('Rounds')
        plt.legend() #设置图例

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        # Plot Buyers profits //买家利润
        plt.figure()
        plt.grid(True, linestyle='--', alpha=0.5)
        for buyer in range(self.n_buyers):
            plt.semilogy(self.cumulative_buyers_profits[buyer], label="Buyer " + str(buyer))

        plt.title('Buyer cumulative profits across all auctions')
        plt.ylabel('Buyer profits')
        plt.xlabel('Rounds')
        plt.legend()

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        #  Plot total profits //总利润
        plt.figure()
        plt.grid(True, linestyle='--', alpha=0.5)
        for auction_round in range(self.r_rounds):
            plt.semilogy(self.cumulative_total_profits)
        plt.title('total cumulative profits across all auctions')
        plt.ylabel('total profits')
        plt.xlabel('Rounds')
        plt.legend()

        if self.r_rounds < 10:
            plt.xticks(range(self.r_rounds))

        plt.show()


if __name__ == '__main__':
    # strategy = [1 for n in range(buyers)]
    # strategy[0] = 4
    auctioneer = Auctioneer1(winner=[], average_winner_bid=[], average_winner_fact_bid=[], starting_prices=[], data=5e5, cycles=10e8, time=0.8, f_m=1e9, Pm=0.3, N_buyers=3, K_sellers=3,
                            R_rounds=10, rE=0.1, rC=60, k_buyers=10e-27, debug=True, w=10e6, Gm=300, n0=10, delay=0.02,Fn=5e9, t=0.001,
                            k_sellers=10e-6, Pn=3)
    auctioneer.start_auction(starting_prices=[])
    auctioneer.plot_statistics()
    print("\n the simulation is finished")
    # auctioneer.print_alphas()
