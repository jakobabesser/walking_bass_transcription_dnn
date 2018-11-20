from __future__ import print_function
import os
import csv
import numpy as np
from scipy.signal import decimate
import soundfile as sf
import librosa
from keras.models import model_from_json

import argparse

__author__ = 'Jakob Abesser'
__copyright__ = 'J. Abesser, S. Balke, K. Frieler, M. Pfleiderer, M. Mueller, 2017'


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
                   dir_out,
                   beat_times=None,
                   tuning_frequency_hz=440.,
                   threshold=0.2,
                   aggregation='beat'):
        """ Transcribe audio file
        Args:
            fn_wav (string): WAV file name
            dir_out (string): Directory to store results (if None, same directory as fn_wav is used)
            beat_times (ndarray): Beat times in seconds (if None, only bass saliency is extracted)
            tuning_frequency_hz (float): Tuning frequency (Hz)
            threshold (float): Decision treshold
            aggregation (string): Aggregation method. Possible values are 'beat' (beat-wise aggregation)"
                                  "and 'flex-q' (dynamic estimation of most likely tatum per beat)
        Returns:
            pitch_saliency (2d ndarray): Bass pitch saliency (num_pitches x num_frames)
            midi_axis (ndarray): MIDI pitch values of pitch axis
            time_axis_sec (ndarray): Time frames [s]
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

        # save saliency matrix
        base_name = os.path.basename(fn_wav).replace('.wav', '')
        np.save(
            os.path.join(dir_out, '{}_bass_pitch_saliency.npy'.format(base_name)),
            pitch_saliency
        )

        # export most salient bass track as CSV file
        with open(os.path.join(dir_out, '{}_bass_f0.csv'.format(base_name)), 'w') as fhandle:
            writer = csv.writer(fhandle, delimiter=',')
            for t in range(pitch_saliency.shape[0]):
                i = np.argmax(pitch_saliency[t])
                time_val = time_axis_sec[t]
                if pitch_saliency[t, i] >= threshold:
                    freq_val = self.f_axis_hz[i]
                else:
                    freq_val = -1*self.f_axis_hz[i]

                writer.writerow([time_val, freq_val])

        # aggregate pitch saliency to note events
        if beat_times is not None:

            onset_sec, offset_sec, pitch = aggregate_saliency_to_notes(pitch_saliency,
                                                                       self.f_axis_midi,
                                                                       time_axis_sec,
                                                                       beat_times,
                                                                       method=aggregation,
                                                                       threshold=threshold)

            # export score to be imported to Sonic Visualiser as note layer
            score_mat = np.vstack((onset_sec, offset_sec, pitch)).T
            np.savetxt(os.path.join(dir_out, '{}_bass_line.csv'.format(base_name)),
                       score_mat,
                       fmt='%4.4f,%4.4f,%d')

        return pitch_saliency, self.f_axis_midi, time_axis_sec


def aggregate_saliency_to_notes(pitch_saliency,
                                freq_bins_midi,
                                frame_times_sec,
                                beat_times_sec,
                                method='beat',
                                num_tatums_per_beat=None,
                                threshold=0.2):
    """ Aggregate frame-wise pitch saliency values to note events based on given beat times
    Args:
        pitch_saliency (2d np.ndarray): Frame-wise pitch saliency (num_frames x num_pitches)
        freq_bins_midi (np.ndarray): MIDI pitch values (num_pitches)
        frame_times_sec (np.ndarray): Frame times in seconds (num_pitches)
        beat_times_sec (np.ndarray): Beat times in seconds (num_beats)
        method (string): Aggregation method
        aggregation (string): Aggregation method. Possible values are 'beat' (beat-wise aggregation)"
                              "and 'flex-q' (dynamic estimation of most likely tatum per beat)
        num_tatums_per_beat (tuple): Number of tatums per beats - different beat subdivisions that are tested if
                                     aggregation == 'flex-q' (default: (1, 2, 3))
        threshold (float): Minimum saliency threshold to detect notes
    Returns:
        onset (np.ndarray): Note-wise onset times in seconds
        offset (np.ndarray): Note-wise offset times in seconds
        pitch (np.ndarray): Note-wise pitch values
    """
    assert method in ('beat', 'flex-q'), "Non-valid value for method!"

    if num_tatums_per_beat is None:
        num_tatums_per_beat = (1, 3)
    num_subdivisions = len(num_tatums_per_beat)

    num_beats = len(beat_times_sec)
    pitch = []
    onset = []
    offset = []

    # map beat times from seconds to frames
    beat_frames = [closest_bin(frame_times_sec, _) for _ in beat_times_sec]

    # iterate over beats
    for b in range(num_beats-1):

        if method == 'beat':
            # take most likely pitch
            beat_pitch_saliency = np.mean(
                pitch_saliency[beat_frames[b]: beat_frames[b + 1], :], axis=0
            )

            # if saliency exceeds threshold > store note
            if np.max(beat_pitch_saliency) > threshold:
                best_pitch_idx = np.argmax(beat_pitch_saliency)
                onset.append(beat_times_sec[b])
                offset.append(beat_times_sec[b+1])
                pitch.append(freq_bins_midi[best_pitch_idx])

        elif method == 'flex-q':

            curr_onset = []
            curr_offset = []
            curr_pitch = []

            scores = np.zeros(num_subdivisions)
            # try different tatum subdivisions
            for s, sub_div in enumerate(num_tatums_per_beat):

                scores[s], _, _ = get_sub_beat_saliency(pitch_saliency,
                                                        beat_times_sec[b],
                                                        beat_times_sec[b + 1],
                                                        frame_times_sec,
                                                        sub_div)

            # get optimal subdivision from highest score
            sub_div_opt = num_tatums_per_beat[np.argmax(scores)]

            # use optimal subdivision to extract note events
            _, sub_beat_saliency, sub_beat_times_sec = get_sub_beat_saliency(pitch_saliency,
                                                                             beat_times_sec[b],
                                                                             beat_times_sec[b + 1],
                                                                             frame_times_sec,
                                                                             sub_div_opt)

            pitch_idx_opt = np.argmax(sub_beat_saliency, axis=1)
            saliency_opt = np.max(sub_beat_saliency, axis=1)

            # check in which tatum segments, the saliency exceeds the threshold
            is_valid = saliency_opt >= threshold

            for n in range(sub_div_opt):
                curr_onset.append(sub_beat_times_sec[n])
                curr_offset.append(sub_beat_times_sec[n+1])
                curr_pitch.append(freq_bins_midi[pitch_idx_opt[n]])

            curr_onset = np.array(curr_onset)
            curr_offset = np.array(curr_offset)
            curr_pitch = np.array(curr_pitch)

            if np.all(is_valid):
                # merge adjacent notes with same pitch as we can't do onset detection solely based on saliency
                if len(np.unique(curr_pitch)) == 1:
                    curr_pitch = np.array((curr_pitch[0],))
                    curr_onset = np.array((curr_onset[0],))
                    curr_offset = np.array((curr_offset[-1],))
            else:
                curr_onset = curr_onset[is_valid]
                curr_offset = curr_offset[is_valid]
                curr_pitch = curr_pitch[is_valid]

            onset.append(curr_onset)
            offset.append(curr_offset)
            pitch.append(curr_pitch)

    if method == 'beat':
        onset = np.array(onset)
        offset = np.array(offset)
        pitch = np.array(pitch)
    elif method == 'flex-q':
        onset = np.concatenate(onset)
        offset = np.concatenate(offset)
        pitch = np.concatenate(pitch).astype(int)

    return onset, offset, pitch


def get_sub_beat_saliency(pitch_saliency,
                          start_time_sec,
                          end_time_sec,
                          frame_times_sec,
                          sub_div):
    """ Get pitch saliency and likelihood score for subdivision of given segment in pitch saliency matrix
    Args:
        pitch_saliency (2d np.ndarray): Frame-wise pitch saliency (num_frames x num_pitches)
        start_time_sec (float): Start time in seconds
        end_time_sec (float): End time in seconds
        frame_times_sec (np.ndarray): Frame times in seconds
        sub_div (int, >= 1): Number of subdivisions (e.g. 2 -> given segment is divided into 2 subsegments of equal
                             duration)
    Returns:
        score (float): Likelihood score for current subdivision based on difference between highest and second
                       highest saliency value
        sub_beat_saliency (2d np.ndarray): Average pitch saliency vectors for each sub beat (tatum level) (num_sub_beats x num_pitches)
        sub_beat_times_sec (np.ndarray): Boundary times in seconds for subbeats (num_sub_beats + 1)
    """
    num_pitches = pitch_saliency.shape[1]
    beat_len_sec = end_time_sec - start_time_sec
    sub_beat_len_sec = beat_len_sec / sub_div
    # sub_beat times
    sub_beat_times_sec = np.arange(sub_div + 1) * sub_beat_len_sec + start_time_sec
    assert len(sub_beat_times_sec) == sub_div + 1
    # seconds to frames
    sub_beat_times_frames = [closest_bin(frame_times_sec, _) for _ in sub_beat_times_sec]
    sub_beat_saliency = np.zeros((sub_div, num_pitches))
    for sb in range(sub_div):
        sub_beat_saliency[sb, :] = get_segment_saliency(pitch_saliency,
                                                        sub_beat_times_frames[sb],
                                                        sub_beat_times_frames[sb + 1])

    # sort pitch saliency values in descending order accross pitches
    sub_beat_saliency_sorted = -np.sort(-sub_beat_saliency, axis=1)
    score = np.mean(sub_beat_saliency_sorted[:, 0] - sub_beat_saliency_sorted[:, 1])

    return score, sub_beat_saliency, sub_beat_times_sec


def get_segment_saliency(pitch_saliency, start_frame, end_frame):
    """ Get average saliency over segment
    Args:
        pitch_saliency (2d np.ndarray): Frame-wise pitch saliency (num_frames x num_pitches)
        start_frame (int): Start frame
        end_frame (int): End frame
    Returns:
        segment_pitch_saliency (np.ndarray): Averaged pitch saliency over segment (num_pitches)
    """
    return np.mean(pitch_saliency[start_frame:end_frame, :], axis=0)


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


def main(args):
    """Main method to run transcription
    """

    # parse beats_file argument
    if args.beats_file == '':
        beat_times = None
    elif not os.path.exists(args.beats_file):
        beat_times = None
        print("[Warning] Could not find provided beats file.")
    else:
        beat_times = np.loadtxt(args.beat_file, delimiter=',', usecols=[0])    

    transcriber = WalkingBassTranscription()

    transcriber.transcribe(args.input_wav,
                           dir_out=args.output_dir,
                           beat_times=beat_times,
                           threshold=args.threshold,
                           aggregation=args.aggregation)

    print('Finished bass line transcription! :)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict walking bass salience or f0"
                    "from and audio file.")
    parser.add_argument("input_wav",
                        type=str,
                        help="Path to input wav file.")
    parser.add_argument("output_dir",
                        type=str,
                        help="Path to save location of bass transcription.")
    parser.add_argument("-b", "--beats_file",
                        type=str,
                        default='',
                        help="Path to beat annotation file. If not given, "
                        "does not produce note-level outputs.")
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=0.2,
                        help="Amplitude threshold. Only used when "
                        "output_format is singlef0 or multif0")
    parser.add_argument("-a", "--aggregation",
                        type=str,
                        default="beat",
                        help="Aggregation method. Possible values are 'beat' (beat-wise aggregation)"
                        "and 'flex-q' (dynamic estimation of most likely tatum per beat)")

    main(parser.parse_args())
