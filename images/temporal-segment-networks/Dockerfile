#https://hub.docker.com/r/bitxiong/tsn/
FROM bitxiong/tsn

RUN	apt-get update && \
	pip install opencv-python && \
	cd /app && \
	bash build_all.sh

COPY ucf101/* /app/models/ucf101/
COPY hmbd51/* /app/models/hmbd51/
COPY train_tsn.sh /app/scripts/train_tsn.sh