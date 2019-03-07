#!/bin/bash
python -c "import bokeh.sampledata; bokeh.sampledata.download()"
bokeh serve --show bokeh_examples/gapminder/main.py
