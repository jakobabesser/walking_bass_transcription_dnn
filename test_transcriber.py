import unittest
import os
import numpy as np

from transcriber import WalkingBassTranscription

""" Simple unit test for transcriber """


class TestTranscriber(unittest.TestCase):

    def setUp(self):
        self.transcriber =  WalkingBassTranscription()
        self.fn_wav = os.path.join('data', 'ArtPepper_Anthropology_Excerpt.wav')
        self.dir_out = 'data'
        # get beat times
        fn_csv = os.path.join('data', 'ArtPepper_Anthropology_Excerpt_beat_times.csv')
        self.beat_times = np.loadtxt(fn_csv, delimiter=',', usecols=[0])

    def test_transcriber(self):
        """ Run transcriber for using different settings """
        for aggregation_method in ('beat', 'flex-q'):
            for threshold in (0, 0.2):
                print('Test for aggregation = {} and threshold = {}'.format(aggregation_method, threshold))

                pitch_saliency, f_axis_midi, time_axis_sec = self.transcriber.transcribe(self.fn_wav,
                                                                                         self.dir_out,
                                                                                         beat_times=self.beat_times,
                                                                                         aggregation=aggregation_method,
                                                                                         threshold=threshold)

                # shape check
                assert pitch_saliency.shape[0] == len(time_axis_sec)
                assert pitch_saliency.shape[1] == len(f_axis_midi)

                # check that result files were generated
                assert os.path.isfile(self.fn_wav.replace('.wav', '_bass_f0.csv'))
                assert os.path.isfile(self.fn_wav.replace('.wav', '_bass_pitch_saliency.npy'))


if __name__ == "__main__":
    unittest.main()
