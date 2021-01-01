import torch

import auction_match_cupy

a = torch.randn(2, 100, 3).cuda()
b = torch.randn(2, 100, 3).cuda()

lmatch, rmatch = auction_match_cupy.AuctionMatch()(a, b)

print(lmatch.size())
print(rmatch.size())

