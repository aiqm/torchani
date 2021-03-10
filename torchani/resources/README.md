# ANI model parameters 

Please note that ANI2x, and ANI1ccx model parameters have been move to https://github.com/aiqm/ani-model-zoo in an effort to reduce the size of TorchANI for deployment.

These parameters will be automatically downloaded once the Built-in classes are called.


### TIPs for Developers
After the parameters are downloaded, git tracked files got modefied. You could run the following command to ignore these changes:
```bash
git update-index --assume-unchanged $(git ls-files | tr '\n' ' ')
```
or ignore changes by the provided script `./assume-unchanged`  and `./no-assume-unchanged` to change it back