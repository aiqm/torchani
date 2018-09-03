FROM zasdfgbnm/pytorch-master
RUN pacman -Sy --noconfirm python-sphinx python2-sphinx python-tqdm python2-tqdm flake8
RUN pip install tensorboardX sphinx-gallery && pip2 install tensorboardX sphinx-gallery
COPY . /torchani
RUN cd torchani && pip install .
RUN cd torchani && pip2 install .
