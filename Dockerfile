FROM zasdfgbnm/pytorch-master
RUN pacman -Sy --noconfirm python-sphinx python2-sphinx python-tqdm python2-tqdm python2-matplotlib python-matplotlib python-pillow python2-pillow flake8
RUN pip install tensorboardX sphinx-gallery ase codecov && pip2 install tensorboardX sphinx-gallery ase codecov
COPY . /torchani
RUN cd torchani && pip install .
RUN cd torchani && pip2 install .
