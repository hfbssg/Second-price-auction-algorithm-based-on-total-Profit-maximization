def start_auction(self, starting_prices):
    """
    Main method of the program, runs the actual simulation //程序的主要方法，运行实际仿真
    """
    self.print_factors()  # 打印拍卖的增减因子
    buyers_bid = {}
    winner_total = 0
    winner_fact_total = 0
    self.auctions_history.append([])
    starting_price = self.starting_prices[0]
    for buyer in range(self.n_buyers):  # 10个买家
        bid = self.calculate_bid(buyer, starting_price)
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