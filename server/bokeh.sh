#!/bin/bash
python -c "import bokeh.sampledata; bokeh.sampledata.download()"
bokeh serve --allow-websocket-origin=localhost:5006 --allow-websocket-origin=0.0.0.0:5006 --show /work/server/bokeh_examples/gapminder/
