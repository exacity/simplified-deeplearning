# Dropout
Simple network with dropout. 

Set train p equal to one means no dropout. It will get an accuracy of about 91%.

Set train p smaller than one means dropout. It will get an accuracy of about 92% when p is equal to 0.95.

Attention! The network is so small that you can't set p too small, otherwise the network will not have enough activation.

It will get an accuracy of about 10% when p is equal to 0.1.

If you want to set p smaller, you need to make a bigger network. 

# Requirement
- Python3
- tensorflow 1.0