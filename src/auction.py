import random
from prettytable import PrettyTable
import numpy as np

class Auction:

    def __init__(self, starting_price, winner, price_paid, bid_history, auction_round, data, cycles, f_m, Pm, N_buyers=3, K_sellers=3, seller_profit=[],
                 rE=0.1, rC=60, Pn=3, Fn=5e9, w=10e6, Gm=300, n0=10, k_buyers=10e-27, delay=0.02, t=0.001, D_loc=[], E_loc=[], D_ori=[],
                 D_coo=[], D_edg=[], E_trans=[], Ori=[], Ocoo=[], Rn=[], k_sellers=10e-6, i=[], winner_profit=[]):
        """
        Object that contains all the useful information of an auction //包含拍卖的所有信息
        :param starting_price: Starting price of the auction
        :param market_price: Market price of the item sold
        :param price_paid: Price to pay for the item
        :param winner: id of the winner of the auction
        :param bid_history: Dictionary with the bids from the buyers that participated in the auction
        :param previous_alphas: Values of the bidding factor applied to calculate the bids of this auction
        :param item_kind: Kind of item sold in this auction
        """
        self.rE = rE
        self.rC = rC
        self.Pn = Pn
        self.Fn = Fn
        self.w = w
        self.Gm = Gm
        self.n0 = n0
        self.delay = delay
        self.t = t
        self.k_n = k_sellers
        self.k_m = k_buyers
        self.N_buyers = N_buyers
        self.K_sellers = K_sellers
        self.auction_round = auction_round
        self.data = data
        self.cycles = cycles
        self.f_m = f_m
        self.Pm = Pm
        self.starting_price = starting_price
        self.price_paid = price_paid
        self.winner = winner
        i.append(random.choice([0, 1]))
        self.i = i[0]
        self.D_loc = self.calculate_D_loc(D_loc)
        self.E_loc = self.calculate_E_loc(E_loc)
        self.Rn = self.calculate_Rn(Rn)
        self.D_ori = self.calculate_D_ori(D_ori)
        self.D_coo = self.calculate_D_coo(D_coo)
        self.D_edg = self.calculate_D_edg(D_edg)
        self.E_trans = self.calculate_E_trans(E_trans)
        self.winner_profit = self.calculate_winner_profit(winner_profit)
        self.Ori = self.calculate_Ori(Ori)
        self.Ocoo = self.calculate_Ocoo(Ocoo)
        self.seller_profit = self.calculate_seller_profit(seller_profit)
        self.item_returned = False

        # Debug purposes
        # ['%.2f' % elem for elem in bid_history.values()]
        self.bid_history = bid_history
        self.new_alphas = []
        self.kept_item_profit = None
        self.kept_item_fee = None
        self.seller_item_kept = None
        self.original_info = None
        self.kept_item_price = None

    #本地卸载
    def calculate_D_loc(self, D_loc):
        """
         Calculate D_loc //计算每个MD本地卸载的时延
        """
        if len(D_loc) > 0:
            return D_loc
        D_loc = []
        D_loc.append(self.cycles / self.f_m)
        return D_loc

    def calculate_E_loc(self, E_loc):
        """
         Calculate E_loc //计算每个MD本地卸载的能耗
        """
        if len(E_loc) > 0:
            return E_loc
        E_loc = []
        E_loc.append(self.k_m * self.cycles * (self.f_m ** 2))
        return E_loc
    #UAV卸载
    def calculate_Rn(self, Rn):
        """
         Calculate Rn 上行传输速率
        """
        if len(Rn) > 0:
            return Rn
        Rn = []
        Rn.append(self.w * np.log(1+(self.Pm * self.Gm) / self.n0))
        return Rn

    def calculate_D_ori(self, D_ori):
        """
         Calculate D_ori //计算原始UAV卸载的时延
        """
        if len(D_ori) > 0:
            return D_ori
        D_ori = []
        D_ori.append(self.data / self.Rn[0] + self.delay + self.cycles / self.Fn)
        return D_ori

    def calculate_D_coo(self, D_coo):
        """
         Calculate D_coo //计算协同UAV卸载的时延
        """
        if len(D_coo) > 0:
            return D_coo
        D_coo = []
        D_coo.append((2 * self.data / self.Rn[0]) + self.delay + (self.cycles / self.Fn))
        return D_coo

    def calculate_D_edg(self, D_edg):
        """
         Calculate D_edg //计算UAV卸载的总时延
        """
        if len(D_edg) > 0:
            return D_edg
        D_edg = []
        D_edg.append((1 - self.i) * self.D_ori[0] + self.i * self.D_coo[0])
        return D_edg

    def calculate_E_trans(self, E_trans):
        """
         Calculate E_trans //计算UAV卸载的总能耗
        """
        if len(E_trans) > 0:
            return E_trans
        E_trans = []
        E_trans.append(self.Pm * (self.data / self.Rn[0]))
        return E_trans
    #MD赢家利润
    def calculate_winner_profit(self, winner_profit):
        """
         Calculate winner_profit //计算赢家实际利润
        """
        if len(winner_profit) > 0:
            return winner_profit
        # winner_profit = np.zeros((auction_round, 1))
        # for auction in range(0, self.auction_round):
        winner_profit = []
        winner_profit.append(self.rE * (self.E_loc[0] - self.E_trans[0]) + self.rC * (self.D_loc[0]- self.D_edg[0] - (self.auction_round+1) * self.t) - 0.8*self.price_paid)
        return winner_profit
    #UAV利润
    def calculate_Ori(self, Ori):
        """
         Calculate Ori 原始UAV的实际利润
        """
        if len(Ori) > 0:
            return Ori
        Ori = []
        Ori.append(self.price_paid - self.rC * ((self.cycles / self.Fn)+(self.auction_round + 1) * self.t) - self.rE * self.k_n * self.data)
        return Ori

    def calculate_Ocoo(self, Ocoo):
        """
         Calculate Ocoo 协作UAV的实际利润
        """
        if len(Ocoo) > 0:
            return Ocoo
        Ocoo = []
        Ocoo.append(self.price_paid - self.rC * ((self.cycles / self.Fn)+(self.auction_round + 1) * self.t) - self.rE * (self.Pn * self.data / self.Rn[0] + self.k_n * self.data))
        return Ocoo

    def calculate_seller_profit(self, seller_profit):
        """
         Calculate seller profit UAV的实际总利润
        """
        if len(seller_profit) > 0:
            return seller_profit
        seller_profit = []
        seller_profit.append((1 - self.i) * self.Ori[0] + self.i * self.Ocoo[0])
        return seller_profit





    def return_item(self, fee, kept_item_profit, kept_item_fee, seller_item_kept, kept_item_price):
        """
        Method used when the item from this auction is returned
        :param fee: fee to pay for returning the item
        :param kept_item_profit: Profit of the item the buyer has decided to keep
        :param kept_item_fee: Fee that the buyer would have payed for canceling the item that is kept
        :param seller_item_kept: Seller of the item kept
        :param kept_item_price: price paid for the item kept
        """
        self.original_info = [self.winner_profit, fee, self.seller_profit]
        self.seller_profit = fee
        self.winner_profit = - fee
        self.item_returned = True
        self.kept_item_profit = kept_item_profit
        self.kept_item_fee = kept_item_fee
        self.kept_item_price = kept_item_price
        self.seller_item_kept = seller_item_kept

    def set_new_alphas(self, new_alphas):
        """
        Save the values of the bidding factors after this auction. Debug and printing purposes mainly. //在这次拍卖之后保存出价因素的值
        :param new_alphas: Values of the new bidding factors
        """
        self.new_alphas = ['%.4f' % elem for elem in new_alphas]
        self.factor = ['%.4f' % (float(new_alpha) / float(old_alpha)) for new_alpha, old_alpha in
                       zip(new_alphas, self.previous_alphas)]

    # def print_auction(self, n):
    #     """
    #     Print the all the information of the auction using tables
    #     :param n: Number of auction (seller id)
    #     """
    #     self.previous_alphas = ['%.4f' % elem for elem in self.previous_alphas]
    #     # Printing buyer info
    #     buyer_info = PrettyTable()
    #     field_names = ["Auction #" + str(n)]
    #     old_alphas = ["Old Alpha"]
    #     new_alphas = ["New Alpha"]
    #     multiplier = ["Multiplier"]
    #     bids = ["Bids"]
    #     for buyer, bid in self.bid_history.items():
    #         heading = "B" + str(buyer)
    #         if buyer == self.winner:
    #             heading = heading + " - W"
    #         field_names.append(heading)
    #         old_alphas.append(self.previous_alphas[buyer])
    #         new_alphas.append(self.new_alphas[buyer])
    #         multiplier.append(self.factor[buyer])
    #         bids.append(round(bid, 2))
    #
    #     buyer_info.field_names = field_names
    #     buyer_info.add_row(old_alphas)
    #     buyer_info.add_row(bids)
    #     buyer_info.add_row(new_alphas)
    #     buyer_info.add_row(multiplier)
    #
    #     print(buyer_info)
    #
    #     # Printing market prices info
    #     auction_info = PrettyTable()
    #
    #     field_names = ["Starting price", "Market Price", "Winner", "Price to pay", "Buyer profit", "Seller profit",
    #                    "Item kind"]
    #     auction_info.field_names = field_names
    #     row = [self.starting_price, self.market_price, self.winner, self.price_paid, self.winner_profit,
    #            self.seller_profit]
    #     row = ['%.2f' % elem for elem in row]
    #     row[2] = self.winner
    #     row.append(self.item_kind)
    #     auction_info.add_row(row)
    #     print(auction_info)
    #
    #     # Printing return info
    #     if self.item_returned:
    #         return_info = PrettyTable()
    #         field_names = ["Buyer profit for discarded item", "Buyer fee for canceling this item",
    #                        "Profit of seller before cancel", "Buyer profit for kept item",
    #                        "Buyer fee if canceling kept item", "Seller of the kept item",
    #                        "Final Profit (profit - fee paid)", "Profit if kept this item"]
    #         return_info.field_names = field_names
    #         row = [self.original_info[0], self.original_info[1],
    #                self.original_info[2], self.kept_item_profit,
    #                self.kept_item_fee, self.seller_item_kept,
    #                self.kept_item_profit - self.original_info[1],
    #                self.original_info[0] - self.kept_item_fee]
    #         row = ['%.2f' % elem for elem in row]
    #         row[5] = self.seller_item_kept
    #         return_info.add_row(row)
    #         print(return_info)
    #     print()
    #     print()

    def round_dict_values(self, dict):
        """
        Round the values of a dictionary to the second decimal
        :param dict: dictionary to round
        :return: dictionary with the new values
        """
        for dict_value in dict:
            for k, v in dict_value.items():
                dict_value[k] = round(v, 2)
        return dict
