def is_toxic(orderbook: dict) -> bool:
    bids = orderbook.get("bids", [])[:3]
    asks = orderbook.get("asks", [])[:3]

    bid_vol = sum([b[1] for b in bids])
    ask_vol = sum([a[1] for a in asks])

    if bid_vol == 0 or ask_vol == 0:
        return True

    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    return abs(imbalance) > 0.6