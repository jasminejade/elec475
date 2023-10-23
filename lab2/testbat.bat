echo Generating output

echo %cd%

python test.py -content_image images/content/cornell.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y

python test.py -content_image images/content/cornell.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y

python test.py -content_image images/content/cornell.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/cornell.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y


python test.py -content_image images/content/airplane.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/Andy_Warhol_97.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y

python test.py -content_image images/content/airplane.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/brushstrokes.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y

python test.py -content_image images/content/airplane.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.9 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.5 -cuda Y
python test.py -content_image images/content/airplane.jpg -style_image images/style/the-persistence-of-memory-1931.jpg -decoder decoder.pth -encoder encoder.pth -alpha 0.1 -cuda Y