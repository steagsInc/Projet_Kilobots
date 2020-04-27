FROM debian:latest
RUN 	apt-get update 
RUN	apt-get install -y  python && \
	apt-get install -y build-essential git gcc-avr gdb-avr binutils-avr avr-libc avrdude libsdl1.2-dev libjansson-dev libsubunit-dev cmake check && \
	git clone https://github.com/JIC-CSB/kilombo.git && \
	cd kilombo && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make install && \
	git clone https://github.com/acornejo/kilolib && \
	cd kilolib && \
	make && \
	apt-get install -y python3-pip && \
	pip3 install cma deap matplotlib scipy opencv-python scikit-learn pandas && \
	apt-get install -y libxrender1 && \
	apt-get install -y jupyter-notebook
RUN cd
RUN mkdir Project
RUN git clone https://github.com/steagsInc/Projet_Kilobots.git
ENTRYPOINT /bin/bash
