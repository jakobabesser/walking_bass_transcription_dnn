import os
import numpy as np
from scipy.signal import decimate
import soundfile as sf
import librosa
from keras.models import model_from_json

__author__ = 'Jakob Abesser'


class WalkingBassTranscription:
    """ Algorithm for walking bass transcription in jazz ensemble recordings
         [1] J. Abesser, S. Balke, K. Frieler, M. Pfleiderer, M. Mueller: Deep Learning for Jazz Walking Bass Transcription, AES conference
             on Semantic Audio, Erlangen, Germany, 2017
        Examples can be found here:
         [2] http://www. audiolabs- erlangen.de/resources/MIR/ 2017-AES-WalkingBassTranscription/
        This algorithm was used to create the beat-wise bass pitch values included in the Weimar Jazz Database
         [3] http://jazzomat.hfm-weimar.de/dbformat/dboverview.html
    """

    def __init__(self,
                 hopsize=1024,
                 blocksize=2048,
                 pitch_range=(28, 67),  # E1 - G4
                 bins_per_octave=12):
        """ Initialize transcriber
        Args:
            hopsize (int): Hopsize in samples
            blocksize (int): Blocksize in samples
            pitch_range (tuple of int): Lower and upper pitch range
            bins_per_octave (int): Frequency axis resolution (number of bins per octave)
        """
        self.hopsize = hopsize
        self.blocksize = blocksize
        self.pitch_range = pitch_range
        self.bins_per_octave = bins_per_octave

        # generate logarithmically spaced frequency axis
        delta_midi = 12. / self.bins_per_octave
        self.f_axis_midi = np.arange(self.pitch_range[0],
                                     self.pitch_range[1] + delta_midi,
                                     delta_midi, dtype=int)
        tuning_freq_hz = 440.
        self.f_axis_hz = tuning_freq_hz * 2 ** ((self.f_axis_midi - 69.) / 12.)

        # load DNN model
        self.model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
        self.model = None
        self._load_model()

    def _load_model(self):
        """ Load DNN model by loading architecture and weights and initialize model accordingly
        """
        fn_model_architecture = os.path.join(self.model_path, 'model.yaml')
        fn_model_weights = os.path.join(self.model_path, 'weights.h5')
        with open(fn_model_architecture, 'r') as f:
            model_json = f.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(fn_model_weights)

    def transcribe(self,
                   fn_wav,
                   dir_out=None,
                   beat_times=None,
                   tuning_frequency_hz=440.):
        """ Transcribe audio file
        Args:
            fn_wav (string): WAV file name
            dir_out (string): Directory to store results (if None, same directory as fn_wav is used)
            beat_times (ndarray): Beat times in seconds (if None, only bass saliency is extracted)
            tuning_frequency_hz (float): Tuning frequency (Hz)
        """
        tuning_dev_in_semitones = np.log2(tuning_frequency_hz/440.)*12

        # load audio file
        x, fs = sf.read(fn_wav)

        # convert to mono
        if x.ndim == 2:
            x = np.mean(x, axis=1)

        # signal decimation
        if fs != 22050.:
            decimation_factor = int(np.round(fs / 22050.))
            x = decimate(x, decimation_factor, zero_phase=True)
            fs /= decimation_factor

        # compute contant-Q spectrogram
        mag_spec = np.abs(librosa.cqt(x,
                                      sr=fs,
                                      hop_length=self.hopsize,
                                      fmin=440*2**((self.pitch_range[0]-69)/12.),
                                      n_bins=self.pitch_range[1] - self.pitch_range[0] + 1,
                                      bins_per_octave=self.bins_per_octave,
                                      tuning=tuning_dev_in_semitones))

        num_frames = mag_spec.shape[1]
        time_axis_sec = (np.arange(num_frames) + .5) * self.hopsize / fs

        # frame stacking
        features = frame_stacking(mag_spec.T, 2)

        # feature normalization
        features = normalize_euclidean(features)

        # model prediction
        pitch_saliency = self.model.predict(features)

        base_name = os.path.basename(fn_wav).replace('.wav', '')
        np.savetxt(os.path.join(dir_out, '{}_bass_pitch_saliency.csv'.format(base_name)),
                                pitch_saliency,
                   delimiter=',',
                   fmt='%4.4f')

        # beat-wise pitch estimation
        if beat_times is not None:
            beat_frames = [closest_bin(time_axis_sec, _) for _ in beat_times]
            num_beats = len(beat_times)-1
            beat_bass_pitch = np.zeros(num_beats)

            for b in range(num_beats):
                beat_pitch_saliency = np.mean(pitch_saliency[beat_frames[b]: beat_frames[b+1], :], axis=0)
                best_pitch_idx = np.argmax(beat_pitch_saliency)
                beat_bass_pitch[b] = self.f_axis_midi[best_pitch_idx]

            score_mat = np.vstack((beat_times[:-1],
                                   beat_times[1:],
                                   beat_bass_pitch)).T

            # export note parameters
            np.savetxt(os.path.join(dir_out, '{}_bass_line.csv'.format(base_name)),
                       score_mat,
                       fmt='%4.4f,%4.4f,%d')


def closest_bin(axis, val):
    """ Return closest bin in axis to value
    Args:
        axis (ndarray): Axis
        val (float / int): Value
    Return:
        idx (int): Closest index in axis to val
    """
    return np.argmin(np.abs(axis-val))


def frame_stacking(x, context_size):
    """ Stack frames to incorporate some temporal context
    Args:
        x (2d ndarray): Feature matrix (num_frames x num_features)
        context_size (int): Context size for frame stacking (e.g. context size of 2 means that we stack 5 frames)
    Return
        x_stacked (2d ndarray): Stacked feature matrix (num_frames x ((context_size*2 + 1)*num_features)
    """
    num_frames = x.shape[0]
    c = 2 * context_size + 1
    assert c < num_frames, "Context size is too big for number of frames! Stacking not possible."
    num_frames_after = num_frames - c
    x_new = []
    for start_frame in range(0, c):
        x_new.append(x[start_frame:start_frame + num_frames_after + 1, :])
    x = np.hstack(x_new)
    x = np.vstack((np.random.random((context_size, x.shape[1])), x, np.random.random((context_size, x.shape[1]))))

    return x


def normalize_euclidean(x):
    """ Frame-wise feature normalization to mean euclidean norm
    Args:
        x (2d ndarray): Feature matrix (num_frames x num_feat_dims)
    Returns:
        x (2d ndarray): Feature matrix (num_frames x num_feat_dims)
    """
    # avoid nans
    idx_zero = np.where(np.sum(x, axis=1) == 0)[0]
    for idx in idx_zero:
        x[idx, :] = .000001
    return x / np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))


if __name__ == '__main__':
    # let's transcribe an example file from the Weimar Jazz Database
    fn_wav = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'ArtPepper_Anthropology_Excerpt.wav')
    fn_csv_beats = fn_wav.replace('.wav', '_beat_times.csv')
    beat_times = np.loadtxt(fn_csv_beats, delimiter=',', usecols=[0])
    transcriber = WalkingBassTranscription()

    transcriber.transcribe(fn_wav,
                           dir_out=os.path.dirname(fn_wav),
                           beat_times=beat_times)

    print('Finished bass line transcription! :)')
