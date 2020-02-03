import json
import math
import os
import re
import warnings
import sys
from configparser import ConfigParser
import numpy as np
import tensorflow as tf
from generator.gpt2.src import encoder, model, sample
from story.utils import *

sys.path.append('Lucidteller/config.ini')
parser = ConfigParser()
parser.read('config.ini')

warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GPT2Generator:
    def __init__(self, generate_num=80, temperature=parser.getfloat('values', 'temp'), top_p=parser.getfloat('values', 'top_p'), censor=False, raw=False, model_name="model_v5"):
        self.generate_num = generate_num
        self.default_gen_num = generate_num
        self.temp = temperature
        #self.top_k = top_k
        self.top_p = top_p
        self.censor = censor
        self.raw = raw
        self.model_name = model_name
        self.model_dir = "generator/gpt2/models"
        self.model_dir = os.path.expanduser(os.path.expandvars(self.model_dir))
        self.word_penalties = dict()

        self.batch_size = 1
        self.samples = 1
        self.vocab = 1

        self.enc = encoder.get_encoder(self.model_name, self.model_dir)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        self.length = tf.placeholder(tf.int32, shape=())
        self.word_penalties_ph = tf.placeholder(tf.float32, [None])
        self.temp_ph = tf.placeholder(tf.float32, shape=())
        self.top_p_ph = tf.placeholder(tf.float32, shape=())
        # np.random.seed(seed)
        # tf.set_random_seed(seed)
        self.gen_output()

        self.saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(self.model_dir, self.model_name))
        self.saver.restore(self.sess, ckpt)

    def prompt_replace(self, prompt):
        # print("\n\nBEFORE PROMPT_REPLACE:")
        # print(repr(prompt))
        prompt = prompt.replace("#", "")
        prompt = prompt.replace("*", "")
        prompt = prompt.replace("\n\n", "\n")
        prompt = re.sub(r"(?<=\w)\.\.(?:\s|$)", ".", prompt)
        prompt = prompt.rstrip(" ")
        # prompt = second_to_first_person(prompt)

        # print("\n\nAFTER PROMPT_REPLACE")
        # print(repr(prompt))
        return prompt

    def result_replace(self, result, actions):
        # print("\n\nBEFORE RESULT_REPLACE:")
        # print(repr(result))

        result = result.replace('."', '".')
        result = result.replace("#", "")
        result = result.replace("*", "")
        result = result.replace("\n\n", "\n")
        result = re.sub(r"(?<=\w)\.\.(?:\s|$)", ".", result)
        # result = first_to_second_person(result)
        result = cut_trailing_sentence(result, self.raw)
        for sentence in actions:
            result = result.replace(sentence.strip()+" ", "")
        if len(result) == 0:
            return ""
        if self.censor:
            result = remove_profanity(result)

        # print("\n\nAFTER RESULT_REPLACE:")
        # print(repr(result))

        return result

    def generate_raw(self, prompt):
        while len(prompt) > 3500:
            prompt = self.cut_down_prompt(prompt)
        context_tokens = self.enc.encode(prompt)
        generated = 0
        for _ in range(self.samples // self.batch_size):
            out = self.sess.run(
                self.output,
                feed_dict={
                    self.context: [context_tokens for _ in range(self.batch_size)],
                    self.context: [context_tokens for _ in range(self.batch_size)],
                    self.length: self.generate_num,
                    self.temp_ph: self.temp,
                    self.top_p_ph: self.top_p,
                    self.word_penalties_ph: [0.0 if penalty is None else penalty for penalty in [self.word_penalties.get(i) for i in range(self.vocab)]]
                },
            )[:, len(context_tokens) :]
            for i in range(self.batch_size):
                generated += 1
                text = self.enc.decode(out[i])
        return text

    def generate(self, prompt, options=None, seed=1, depth=1):

        debug_print = False
        prompt = self.prompt_replace(prompt)
        last_prompt = prompt[prompt.rfind(">")+2:] if prompt.rfind(">") > -1 else prompt

        if debug_print:
            print("******DEBUG******")
            print("Prompt is: ", repr(prompt))

        text = self.generate_raw(prompt)

        if debug_print:
            print("Generated result is: ", repr(text))
            print("******END DEBUG******")

        result = text
        result = self.result_replace(result, re.findall(r".+?(?:\.{1,3}|[!\?]|$)(?!\")", last_prompt))
        if len(result) == 0 and depth < 20:
            return self.generate(self.cut_down_prompt(prompt), depth=depth+1)
        elif result.count(".") < 2 and depth < 20:
            return self.generate(prompt, depth=depth+1)

        return result

    def cut_down_prompt(self, prompt):
        if not self.raw:
            split_prompt = prompt.split(">")
            expendable_text = ">".join(split_prompt[2:])
            return split_prompt[0] + (">" + expendable_text if len(expendable_text) > 0 else "")
        else:
            sentences = string_to_sentence_list(prompt.lstrip())
            sentences = sentences[1:]
            new_text = ""
            for i in range(len(sentences)):
                if sentences[i] == "<break>":
                    new_text = new_text + "\n"
                else:
                    new_text = new_text + " " + sentences[i]
            return new_text.lstrip()

    def gen_output(self):
        models_dir = os.path.expanduser(os.path.expandvars(self.model_dir))
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, self.model_name, "hparams.json")) as f:
            hparams.override_from_dict(json.load(f))
        self.vocab = hparams.n_vocab
        seed = np.random.randint(0, 100000)
        self.output = sample.sample_sequence(
            hparams=hparams,
            length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temp_ph,
            #top_k=self.top_k,
            top_p=self.top_p_ph,
            word_penalties=self.word_penalties_ph,
        )

    def change_temp(self, t):
        self.temp = t

    def change_top_p(self, t):
        self.top_p = t

    def change_raw(self, raw):
        self.raw = raw

    def set_word_penalties(self, word_penalties):
        self.word_penalties=dict(parser[penalties])
        for token, index in self.enc.encoder.items():
            for word, penalty in word_penalties.items():
                if re.search(word, token, re.IGNORECASE) is not None:
                    self.word_penalties[index] = float(penalty/math.log2(math.e))
