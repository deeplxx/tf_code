import urllib.request
import zipfile
import os
import tensorflow as tf
import collections
import numpy as np


def maybe_download(url, filename, expected_bytes=None):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)  # 获取一个资源并且存储在某个临时的空间

    # statinfo = os.startfile(filename)  # 打开文件
    # if statinfo.st_size == expected_bytes:
    #     print('Found and verified', filename)
    # else:
    #     print(statinfo.st_size)
    #     raise Exception('Fauled to verify')

    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(vocabulary_size, words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
        """data是索引列表，一维"""

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def creat_data():
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download(url, 'text8.zip', 31344016)
    words = read_data(filename)
    print('Data size', len(words))

    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary_size, words)
    del words
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    return data, count, dictionary, reverse_dictionary


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    data_index = 0
    data, _, _, _ = creat_data()
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


if __name__ is '__main__':
    batch, label = generate_batch(8, 2, 1)
    print(batch, '\n', label)
