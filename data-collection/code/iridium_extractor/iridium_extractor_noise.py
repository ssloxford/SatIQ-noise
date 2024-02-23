#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Iridium Extractor
# GNU Radio version: v3.9.6.0-1-gac58f3ba

from gnuradio import gr
from gnuradio import blocks
from gnuradio import analog
import pmt
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
from gnuradio import soapy
import iridium
import math

import iq_out
import decoded_out
import debug_print
import pdu_split
import signal_strength


class iridium_extractor(gr.top_block):

    def __init__(self, zmq_address_iq, zmq_address_bytes, verbose=False, in_file=None, bandwidth=8000000, burst_threshold=26, burst_post_len=32000, burst_pre_len=4096, burst_sample_rate=2000000, burst_width=40000, center_freq=1626000000, gain=80, max_burst_len=180000, sample_rate=2000000, num_samples=None, parallelism=1, noise=0, tx_bandwidth=8000000, tx_freq=1623500000, tx_gain=45, tx_samp_rate=8000000):
        gr.top_block.__init__(self, "Iridium Extractor", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.in_file = in_file
        self.bandwidth = bandwidth
        self.burst_threshold = burst_threshold
        self.burst_post_len = burst_post_len
        self.burst_pre_len = burst_pre_len
        self.burst_sample_rate = burst_sample_rate
        self.burst_width = burst_width
        self.center_freq = center_freq
        self.gain = gain
        self.max_burst_len = max_burst_len
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.parallelism = parallelism
        self.verbose = verbose or False

        self.noise = noise
        self.tx_bandwidth = tx_bandwidth
        self.tx_freq = tx_freq
        self.tx_gain = tx_gain
        self.tx_samp_rate = tx_samp_rate

        print("          NUM_SAMPLES: {}".format(self.num_samples))
        print("      BURST_THRESHOLD: {}".format(self.burst_threshold))
        print("                NOISE: {}".format(self.noise))
        print("         TX BANDWIDTH: {}".format(self.tx_bandwidth))
        print("              TX GAIN: {}".format(self.tx_gain))
        print("              TX FREQ: {}".format(self.tx_freq))
        print("       TX SAMPLE RATE: {}".format(self.tx_samp_rate))

        ##################################################
        # Variables
        ##################################################
        self.start_finder_filter = start_finder_filter = firdes.low_pass(1.0, burst_sample_rate, 5e3 / 2,10e3 / 2, window.WIN_HAMMING, 6.76)
        self.max_queue_len = max_queue_len = 500
        self.input_filter = input_filter = firdes.low_pass(1.0, sample_rate, burst_width / 2,burst_width, window.WIN_HAMMING, 6.76)
        self.handle_multiple_frames_per_burst = handle_multiple_frames_per_burst = False
        self.fft_size = fft_size = 2**round(math.log(sample_rate / 1000, 2))
        ##################################################
        # Blocks
        ##################################################

        if self.in_file is None:
            self.uhd_usrp_source_0 = uhd.usrp_source(
                #",".join(("addr=192.168.10.2", '')),
                ",".join(("")),
                uhd.stream_args(
                    cpu_format="fc32",
                    args='',
                    channels=list(range(0,1)),
                ),
            )
            self.uhd_usrp_source_0.set_samp_rate(sample_rate)
            # No synchronization enforced.

            self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
            self.uhd_usrp_source_0.set_antenna("RX2", 0)
            self.uhd_usrp_source_0.set_bandwidth(bandwidth, 0)
            self.uhd_usrp_source_0.set_gain(gain, 0)
        else:
            self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, self.in_file, False, 0, 0)
            self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)

        self.iridium_tagged_burst_to_pdu_0 = iridium.tagged_burst_to_pdu(max_burst_len, 0.0, 1.0, 1.0, 0, max_queue_len, False)
        self.iridium_fft_burst_tagger_0 = iridium.fft_burst_tagger(center_freq, fft_size, sample_rate, burst_pre_len, burst_post_len,
                                    burst_width, 0, burst_threshold, 512, False)

        self.iridium_burst_downmix = []
        self.iridium_qpsk_demod = []
        for i in range(self.parallelism):
            self.iridium_burst_downmix.append(iridium.burst_downmix(burst_sample_rate, int(0.007 * burst_sample_rate), 0, input_filter, start_finder_filter, handle_multiple_frames_per_burst))
            self.iridium_qpsk_demod.append(iridium.iridium_qpsk_demod(1))

        self.iq_out = iq_out.blk(zmq_address_iq, num_samples=self.num_samples, debug=False)
        self.decoded_out = decoded_out.blk(zmq_address_bytes, debug=False)

        self.pdu_split = pdu_split.blk(num_outputs=self.parallelism)

        self.debug = debug_print.blk()

        if self.verbose:
            self.iridium_iridium_frame_printer_0 = iridium.iridium_frame_printer("")
            self.iridium_frame_sorter_0 = iridium.frame_sorter()

        self.soapy_bladerf_sink_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_sink_0 = soapy.sink(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_sink_0.set_sample_rate(0, tx_samp_rate)
        self.soapy_bladerf_sink_0.set_bandwidth(0, tx_bandwidth)
        self.soapy_bladerf_sink_0.set_frequency(0, tx_freq)
        self.soapy_bladerf_sink_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_sink_0.set_gain(0, min(max(tx_gain, 17.0), 73.0))
        self.blocks_add_const_vxx_0 = blocks.add_const_cc(0.0105+0.0048j)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, (noise**2), 0)

        ##################################################
        # Connections
        ##################################################
        if self.in_file is None:
            self.connect((self.uhd_usrp_source_0, 0), (self.iridium_fft_burst_tagger_0, 0))
            #self.connect((self.uhd_usrp_source_0, 0), (self.signal_strength_0, 0))
            #self.connect((self.signal_strength_0, 0), (self.null_0, 0))
        else:
            self.connect((self.blocks_file_source_0, 0), (self.iridium_fft_burst_tagger_0, 0))

        self.connect((self.iridium_fft_burst_tagger_0, 0), (self.iridium_tagged_burst_to_pdu_0, 0))

        self.msg_connect((self.iridium_tagged_burst_to_pdu_0, 'cpdus'), (self.pdu_split, 'in'))
        for i in range(self.parallelism):
            self.msg_connect((self.pdu_split, 'out{}'.format(i)), (self.iridium_burst_downmix[i], 'cpdus'))
            #self.msg_connect((self.pdu_split, 'out{}'.format(i)), (self.iridium_header_extract[i], 'cpdus'))
            self.msg_connect((self.iridium_burst_downmix[i], 'burst_handled'), (self.iridium_tagged_burst_to_pdu_0, 'burst_handled'))
            self.msg_connect((self.iridium_burst_downmix[i], 'cpdus'), (self.iridium_qpsk_demod[i], 'cpdus'))

            self.msg_connect((self.iridium_burst_downmix[i], 'cpdus'), (self.iq_out, 'pdus'))
            #self.msg_connect((self.iridium_header_extract[i], 'cpdus'), (self.iq_out, 'pdus'))
            self.msg_connect((self.iridium_qpsk_demod[i], 'pdus'), (self.decoded_out, 'pdus'))

        self.msg_connect((self.iridium_tagged_burst_to_pdu_0, 'cpdus'), (self.debug, 'pdus_a'))
        for i in range(self.parallelism):
            self.msg_connect((self.iridium_burst_downmix[i], 'cpdus'), (self.debug, 'pdus_b'))
            #self.msg_connect((self.iridium_header_extract[i], 'cpdus'), (self.debug, 'pdus_b'))
            self.msg_connect((self.iridium_qpsk_demod[i], 'pdus'), (self.debug, 'pdus_c'))

        if self.verbose:
            self.msg_connect((self.iridium_iridium_qpsk_demod_0, 'pdus'), (self.iridium_frame_sorter_0, 'pdus'))
            self.msg_connect((self.iridium_frame_sorter_0, 'pdus'), (self.iridium_iridium_frame_printer_0, 'pdus'))

        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.soapy_bladerf_sink_0, 0))


    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.uhd_usrp_source_0.set_bandwidth(self.bandwidth, 0)

    def get_burst_post_len(self):
        return self.burst_post_len

    def set_burst_post_len(self, burst_post_len):
        self.burst_post_len = burst_post_len

    def get_burst_pre_len(self):
        return self.burst_pre_len

    def set_burst_pre_len(self, burst_pre_len):
        self.burst_pre_len = burst_pre_len

    def get_burst_sample_rate(self):
        return self.burst_sample_rate

    def set_burst_sample_rate(self, burst_sample_rate):
        self.burst_sample_rate = burst_sample_rate
        self.set_start_finder_filter(firdes.low_pass(1.0, self.burst_sample_rate, 5e3 / 2, 10e3 / 2, window.WIN_HAMMING, 6.76))

    def get_burst_width(self):
        return self.burst_width

    def set_burst_width(self, burst_width):
        self.burst_width = burst_width
        self.set_input_filter(firdes.low_pass(1.0, self.sample_rate, self.burst_width / 2, self.burst_width, window.WIN_HAMMING, 6.76))

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_max_burst_len(self):
        return self.max_burst_len

    def set_max_burst_len(self, max_burst_len):
        self.max_burst_len = max_burst_len

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.set_fft_size(2**round(math.log(self.sample_rate / 1000, 2)))
        self.set_input_filter(firdes.low_pass(1.0, self.sample_rate, self.burst_width / 2, self.burst_width, window.WIN_HAMMING, 6.76))
        self.uhd_usrp_source_0.set_samp_rate(self.sample_rate)

    def get_start_finder_filter(self):
        return self.start_finder_filter

    def set_start_finder_filter(self, start_finder_filter):
        self.start_finder_filter = start_finder_filter

    def get_max_queue_len(self):
        return self.max_queue_len

    def set_max_queue_len(self, max_queue_len):
        self.max_queue_len = max_queue_len

    def get_input_filter(self):
        return self.input_filter

    def set_input_filter(self, input_filter):
        self.input_filter = input_filter

    def get_handle_multiple_frames_per_burst(self):
        return self.handle_multiple_frames_per_burst

    def set_handle_multiple_frames_per_burst(self, handle_multiple_frames_per_burst):
        self.handle_multiple_frames_per_burst = handle_multiple_frames_per_burst

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--zmq-address-iq', dest='zmq_address_iq', required=True, help='ZeroMQ address for iq samples')
    parser.add_argument('--zmq-address-bytes', dest='zmq_address_bytes', required=True, help='ZeroMQ address for bytes')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', help='Verbose output')
    parser.add_argument(
        "--in-file", dest="in_file", type=str, default=None,
        help="Set Source File (uses USRP direct input if left empty) [default=%(default)r]")
    parser.add_argument(
        "--bandwidth", dest="bandwidth", type=intx, default=8000000,
        help="Set Bandwidth (Hz) [default=%(default)r]")
    parser.add_argument(
        "--burst-threshold", dest="burst_threshold", type=intx, default=26,
        help="Set Burst Tagging Threshold (dB) [default=%(default)r]")
    parser.add_argument(
        "--burst-post-len", dest="burst_post_len", type=intx, default=32000,
        help="Set Post Burst Samples [default=%(default)r]")
    parser.add_argument(
        "--burst-pre-len", dest="burst_pre_len", type=intx, default=4096,
        help="Set Pre Burst Samples [default=%(default)r]")
    parser.add_argument(
        "--burst-sample-rate", dest="burst_sample_rate", type=intx, default=2000000,
        help="Set Burst Sample Rate (Sps) [default=%(default)r]")
    parser.add_argument(
        "--burst-width", dest="burst_width", type=intx, default=40000,
        help="Set Burst Width [default=%(default)r]")
    parser.add_argument(
        "--center-freq", dest="center_freq", type=intx, default=1626000000,
        help="Set Center Frequency (Hz) [default=%(default)r]")
    parser.add_argument(
        "--gain", dest="gain", type=eng_float, default=eng_notation.num_to_str(float(80)),
        help="Set Gain (dB) [default=%(default)r]")
    parser.add_argument(
        "--max-burst-len", dest="max_burst_len", type=intx, default=180000,
        help="Set Max Burst Size [default=%(default)r]")
    parser.add_argument(
        "--sample-rate", dest="sample_rate", type=intx, default=2000000,
        help="Set Sample Rate (Sps) [default=%(default)r]")
    parser.add_argument(
        "--num-samples", dest="num_samples", type=intx, default=None,
        help="Set Number of Burst Samples [default=%(default)r]")
    parser.add_argument(
        "--parallelism", dest="parallelism", type=intx, default=1,
        help="Set Number of Threads for Processing [default=%(default)r]")
    parser.add_argument(
        "--noise", dest="noise", type=float, default=0,
        help="Set Tx noise [default=%(default)r]")
    parser.add_argument(
        "--tx-bandwidth", dest="tx_bandwidth", type=intx, default=8000000,
        help="Set Tx bandwidth [default=%(default)r]")
    parser.add_argument(
        "--tx-freq", dest="tx_freq", type=intx, default=1623500000,
        help="Set Tx frequency [default=%(default)r]")
    parser.add_argument(
        "--tx-gain", dest="tx_gain", type=intx, default=45,
        help="Set Tx gain [default=%(default)r]")
    parser.add_argument(
        "--tx-samp-rate", dest="tx_samp_rate", type=intx, default=8000000,
        help="Set Tx sample rate [default=%(default)r]")
    return parser


def main(top_block_cls=iridium_extractor, options=None):
    if options is None:
        options = argument_parser().parse_args()

    print("          SAMPLE RATE: {}".format(options.sample_rate))
    print("    BURST SAMPLE RATE: {}".format(options.burst_sample_rate))

    # Try creating tb twice to catch any exceptions
    for i in range(2):
        try:
            tb = top_block_cls(zmq_address_iq=options.zmq_address_iq, zmq_address_bytes=options.zmq_address_bytes, verbose=options.verbose, in_file=options.in_file, bandwidth=options.bandwidth, burst_threshold=options.burst_threshold, burst_post_len=options.burst_post_len, burst_pre_len=options.burst_pre_len, burst_sample_rate=options.burst_sample_rate, burst_width=options.burst_width, center_freq=options.center_freq, gain=options.gain, max_burst_len=options.max_burst_len, sample_rate=options.sample_rate, num_samples=options.num_samples, parallelism=options.parallelism, noise=options.noise, tx_bandwidth=options.tx_bandwidth, tx_freq=options.tx_freq, tx_gain=options.tx_gain, tx_samp_rate=options.tx_samp_rate)
            break
        except Exception as e:
            print(e)
            if i == 1:
                raise

    #tb = top_block_cls(zmq_address_iq=options.zmq_address_iq, zmq_address_bytes=options.zmq_address_bytes, bandwidth=options.bandwidth, burst_post_len=options.burst_post_len, burst_pre_len=options.burst_pre_len, burst_sample_rate=options.burst_sample_rate, burst_width=options.burst_width, center_freq=options.center_freq, gain=options.gain, max_burst_len=options.max_burst_len, sample_rate=options.sample_rate)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
