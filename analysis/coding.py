import numpy as np
from scipy import special
import operator

# Given a bit error rate, what's the probability that the packet doesn't decode?

def bch_error_rate(ber, data_length, parity_length, t, interleave_depth, symbol_size):
    symbol_correct_rate = (1 - ber)**symbol_size
    symbol_error_rate = 1 - (1 - ber)**symbol_size
    block_length = (data_length+parity_length)/interleave_depth

    block_correct_rate = np.sum(list(map(
        lambda e: (
            (symbol_correct_rate)**(block_length-e)
            * symbol_error_rate**e
            * special.comb(block_length, e)
        ),
        range(0, t+1)
    )))
    packet_error_rate = 1 - (block_correct_rate)**interleave_depth
    len = (data_length + parity_length)*symbol_size*interleave_depth
    equivalent_ber = packet_error_rate*0.5
    return {
        "equivalent_ber": equivalent_ber,
        "packet_error_rate": packet_error_rate
    }

def rs_error_rate(ber, data_length, parity_length, interleave_depth=1, symbol_size=8):
    symbol_correct_rate = (1 - ber)**symbol_size
    symbol_error_rate = 1 - (1 - ber)**symbol_size
    block_length = (data_length+parity_length)/interleave_depth
    print(f"block length: {block_length}")
    corruptible_symbols = int(parity_length/interleave_depth/2) # the number of corrupted symbols that can be corrected
    print(f"corruptible_symbols: {corruptible_symbols}")

    print(list(map(
        lambda e: (
        (symbol_correct_rate)**(block_length-e)
        * symbol_error_rate**e
        * special.comb(block_length, e)
        ),
        range(0, corruptible_symbols+1)
    )))

    block_correct_rate = np.sum(list(map(
        lambda e: (
            (symbol_correct_rate)**(block_length-e)
            * symbol_error_rate**e
            * special.comb(block_length, e)
        ),
        range(0, corruptible_symbols+1)
    )))
    print(f"block correct rate: {block_correct_rate}")


    block_error_rate = 1 - np.sum(list(map(
        lambda e: (
            (symbol_correct_rate)**(block_length-e)
            * symbol_error_rate**e
            * special.comb(block_length, e)
        ),
        range(0, corruptible_symbols+1)
    )))

    packet_correct_rate = (block_correct_rate)**interleave_depth
    packet_error_rate = 1 - (block_correct_rate)**interleave_depth
    # equivalent_ser is the ser we'd get without the code check
    len = (data_length + parity_length)*symbol_size*interleave_depth
    equivalent_ber = 1 - (packet_correct_rate)**(1/len)
    eq_ber_2 = packet_error_rate * 0.5

    return {
        "symbol_error_rate": symbol_error_rate,
        "block_correct_rate": block_correct_rate,
        "packet_correct_rate": packet_correct_rate,
        "equivalent_ber": eq_ber_2,
        "improvement rate": ber/equivalent_ber
    }

def rs_bit_error_rate(ber, data_length, parity_length, interleave_depth=1, symbol_size=8):
    return rs_error_rate(ber, data_length, parity_length, interleave_depth, symbol_size)["equivalent_ber"]


def hamming_7_4(ber):
    # symbol error rate

    p0 = (1-ber)**7
    p1 = (1-ber)**6 * ber * 7
    return 0.5 * (1 - (p0 + p1))

def soft_decoding_bonus(x):
    # TODO: actual soft decisions
    return x-2

# TODO: convolutional coding
# TODO: concatenated coding

# convolutional coding

# from "FEC Performance of Concatenated Reed-Solomon and Convolutional Coding with Interleaving"

conv_params = {
    (1,2): {
        "d_free": 10,
        "c_d": [36, 0, 211, 0, 1404, 0, 11633, 0, 77433, 0, 502690, 0,3322763, 0, 21292910, 0, 134365911, 0],
        "aggregate_code_rate": 0.4608
    },
    (2,3): {
        "d_free": 6,
        "c_d": [3,70,285, 1276,6160, 27128,117019],
        "aggregate_code_rate": 0.6144
    },
    (3,4): {
        "d_free": 5,
        "c_d": [42,201,1492,10469,62935,379644],
        "aggregate_code_rate": 0.6912
    },
    (5,6): {
        "d_free": 4,
        "c_d": [92,528,8694,79453,792114],
        "aggregate_code_rate": 0.768
    },
    (7,8): {
        "d_free": 3,
        "c_d": [9,500,7437,105707,1402743],
        "aggregate_code_rate": 0.8064
    }
}

# ber_func maps code rate to ber
def conv_error_rate(ber_func, n, k):
    d_free = conv_params[(n,k)]["d_free"]
    c_d = conv_params[(n,k)]["c_d"]
    aggregate_code_rate = conv_params[(n,k)]["aggregate_code_rate"]

    #rates = list(map(lambda x: x*(n/k), list(range(d_free, d_free + len(c_d)))))
    rates = list(map(lambda x: x*aggregate_code_rate, list(range(d_free, d_free + len(c_d)))))

    P_d = list(map(ber_func, rates))
    print("###")
    print(f"rates: {rates}")
    #exit(1)
    print(f"c_d: {c_d}")

    l = []
    for i in range(len(P_d[0])):
        l2 = [p[i] for p in P_d]
        l.append((1/n) * sum(map(operator.mul, c_d, l2)))

    print(f"l: {list(l)}")

    return l

    #(1/k) * sum(map(operator.mul, c_d, P_d))
