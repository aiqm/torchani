FROM zasdfgbnm/pytorch-master
RUN pacman -Sy --noconfirm python-sphinx python2-sphinx
COPY . /torchani
RUN cd torchani && pip install .
RUN cd torchani && pip2 install .
