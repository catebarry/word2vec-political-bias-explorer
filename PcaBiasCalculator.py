import re
from gensim.models import KeyedVectors
from os import path
import numpy as np
from sklearn.decomposition import PCA

# can adjust pairs, experimented to have a larger spread between left and right
political_biased_word_pairs = [
    ("democrat", "republican"),
    ("democrats", "republicans"),
    ("liberal", "conservative"),
    ("Dems", "GOP"),
    ("CNN", "Fox"),
    ("Pelosi", "McConnell"),
    ("Obama", "Bush"),
    ("investigation", "hoax"),
    ("refugee", "illegal"),
    #("MSNBC", "Fox"),
    #("socialism", "capitalism"),
    #("Clinton", "Cheney"),
    #("progressive", "conservative"),
    #("blue", "red"),
    #("left", "right"),
    #("immigration", "invasion"),
    #("programs", "entitlements"),
]

political_neutral_words = [
    "is",
    "who",
    "what",
    "where",
    "the",
    "it",
]


class PcaBiasCalculator:
    def __init__(
        self,
        model_path=path.join(
            path.dirname(__file__), "data/GoogleNews-vectors-negative300.bin"
        ),
        biased_word_pairs=political_biased_word_pairs,
        neutral_words=political_neutral_words,
    ):
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.biased_word_pairs = biased_word_pairs
        self.neutral_words = neutral_words

        biased_pairs = [
            (self.model[pair[0]], self.model[pair[1]]) for pair in biased_word_pairs
        ]
        biases = [pair[0] - pair[1] for pair in biased_pairs]
        reversed_biases = [pair[1] - pair[0] for pair in biased_pairs]

        self.pca = PCA(n_components=1)
        self.pca.fit(np.array(biases + reversed_biases))

        left_mean = np.mean(self.pca.transform(np.array([pair[0] for pair in biased_pairs])))
        right_mean = np.mean(self.pca.transform(np.array([pair[1] for pair in biased_pairs])))
        neutral_mean = np.mean(self.pca.transform(np.array([self.model[word] for word in neutral_words])))

        # The following 10 lines were modified with help from ChatGPT 5.0 (see AI Assistance Statement)
        shift = neutral_mean
        left_mean -= shift
        right_mean -= shift
        self.left_mean = left_mean
        self.right_mean = right_mean
        self.neutral_mean = 0.0

        #self.positive_mean = max(right_mean, left_mean)
        #self.negative_mean = min(right_mean, left_mean)
        self.sign = 1 if right_mean > left_mean else -1
        EPS_DIST = 0.05 # ADJUST
        self.pos_dist = max(EPS_DIST, abs(self.right_mean))
        self.neg_dist = max(EPS_DIST, abs(self.left_mean))

        # debug
        #print(f"CALIBRATION: left_mean={self.left_mean:.6f} right_mean={self.right_mean:.6f} pos_dist={self.pos_dist:.6f} neg_dist={self.neg_dist:.6f} sign={self.sign}")

    def keys(self):
        return self.model.key_to_index.keys()


    def detect_bias(self, raw_word):
        """
        Use PCA to find the political bias vector, and determine bias based on position along the political vector
        """
        word = re.sub(r"\s+", "_", raw_word)
        if word not in self.model:
            return None
        
        word_val = self.pca.transform(np.array([self.model[word]]))[0][0]

        # The following 3 lines were modified with help from ChatGPT 5.0 (see AI Assistance Statement)
        # rescaling word value so that the left/right average maps to 1 and -1, and neutral_mean maps to 0
        denom = self.pos_dist if word_val > 0 else self.neg_dist
        raw = word_val / denom
        return float(self.sign * raw)
        '''
        if word_val > self.neutral_mean:
            return self.sign * float(
                (word_val - self.neutral_mean)
                / (self.positive_mean - self.neutral_mean)
            )
        else:
            return self.sign * float(
                (self.neutral_mean - word_val)
                / (self.negative_mean - self.neutral_mean)
            )
        '''
