import os
import shutil
from multiprocessing import Process

import click
from captcha.image import ImageCaptcha
from tqdm import tqdm


@click.command(help="Generate captcha images, based on words from dictionary file.")
@click.option("--width", type=click.INT, default=160, help="Width of generated image.")
@click.option("--height", type=click.INT, default=60, help="Height of generated image.")
@click.option("--dict-path", type=click.STRING,
              default="generator/google-10000-english-master/google-10000-english-usa-no-swears-medium.txt",
              help="Path of dict containing words, which will be generated.")
@click.option("--n-words", type=click.INT, default=100,
              help="Number of different words, which will be generated. Starts taking from top of the file.")
@click.option("--duplicates", type=click.INT, default=1000,
              help="Number of duplicates of the same captcha word (but image will be slightly different.")
@click.option("--output-dir", type=click.STRING, default="output",
              help="Output dir, where captchas will be generated to.")
def generate(width, height, dict_path, n_words, duplicates, output_dir):
    image = ImageCaptcha(width=width, height=height, font_sizes=[44])

    with open(dict_path, 'r') as source:
        words = source.readlines()

        try:
            shutil.rmtree(output_dir)
        except:
            pass
        os.makedirs(output_dir)

    for word in tqdm(words[:n_words]):
        word = word.replace('\n', '')
        cpus = os.cpu_count()
        ps = [Process(target=generate_captchas,
                      args=(image, word, output_dir, int(cpu / cpus * duplicates), int((cpu + 1) / cpus * duplicates)))
              for cpu in range(cpus)]

        for p in ps:
            p.start()

        for p in ps:
            p.join()


def generate_captchas(image, word, output_dir, d_from, d_to):
    for i in range(d_from, d_to):
        image.write(word, os.path.join(output_dir, '{}_{}.png'.format(word, i)))


if __name__ == '__main__':
    generate()
