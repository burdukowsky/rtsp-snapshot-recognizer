import datetime
import os
import vlc
import time
import tensorflow as tf
import numpy as np

snapshots_path = 'snapshots/'


def get_snapshot_path():
    return snapshots_path + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3] + '.png'


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    model_file = "network/output_graph.pb"
    label_file = "network/output_labels.txt"

    input_layer = "Placeholder"
    output_layer = "final_result"

    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255

    if not os.path.exists(snapshots_path):
        os.mkdir(snapshots_path)

    player = vlc.MediaPlayer("rtsp://192.168.0.85:5554/camera")
    player.play()

    while 1:
        time.sleep(1)
        snapshot_path = get_snapshot_path()
        take_snapshot_result = player.video_take_snapshot(0, snapshot_path, 0, 0)
        if take_snapshot_result != 0:
            print('RTSP snapshot error. Result = %s' % take_snapshot_result)
            continue

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(
            snapshot_path,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.compat.v1.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-2:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
            print(labels[i], results[i])
        print('\n\n')

        os.remove(snapshot_path)
