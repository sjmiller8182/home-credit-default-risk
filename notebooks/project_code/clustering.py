import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model

def plot_tsne_clusters(data, 
                       labels, 
                       perplexity = 30,
                       s = 0.5, 
                       figsize = (10,10),
                       random_state = None):
    """Plot tsne 2D composition
    """
    # calculate tsne
    tsne = TSNE(random_state = random_state,
                perplexity = perplexity)
    tsne_comp = tsne.fit_transform(data)
    
    # create plot
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))
    plt.figure(figsize = figsize)
    plt.scatter(tsne_comp[:, 0], tsne_comp[:, 1],
                c = palette[labels], s = s);
    
class AutoEncoder:
    """AutoeEncoder: standard feed forward autoencoder

    Parameters:
    -----------
    input_dim: int
        The number of dimensions of your input


    latent_dim: int
        The number of dimensions which you wish to represent the data as.

    architecture: list
        The structure of the hidden architecture of the networks. for example,
        the n2d default is [500, 500, 2000],
        which means the encoder has the structure of:
        [input_dim, 500, 500, 2000, latent_dim], and the decoder has the structure of:
        [latent_dim, 2000, 500, 500, input dim]

    act: string
        The activation function. Defaults to 'relu'
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        architecture=[500, 500, 2000],
        act="relu",
        x_lambda=lambda x: x,
    ):
        shape = [input_dim] + architecture + [latent_dim]
        self.x_lambda = x_lambda
        self.dims = shape
        self.act = act
        self.x = Input(shape=(self.dims[0],), name="input")
        self.h = self.x
        n_stacks = len(self.dims) - 1
        for i in range(n_stacks - 1):
            self.h = Dense(
                self.dims[i + 1], activation=self.act, name="encoder_%d" % i
            )(self.h)
        self.encoder = Dense(self.dims[-1], name="encoder_%d" % (n_stacks - 1))(self.h)
        self.decoded = Dense(self.dims[-2], name="decoder", activation=self.act)(
            self.encoder
        )
        for i in range(n_stacks - 2, 0, -1):
            self.decoded = Dense(
                self.dims[i], activation=self.act, name="decoder_%d" % i
            )(self.decoded)
        self.decoded = Dense(self.dims[0], name="decoder_0")(self.decoded)

        self.Model = Model(inputs=self.x, outputs=self.decoded)
        self.encoder = Model(inputs=self.x, outputs=self.encoder)
        self.__hist = None

    def fit(
        self,
        x,
        batch_size,
        epochs,
        loss,
        optimizer,
        weights,
        verbose,
        weight_id,
        patience,
    ):
        """fit: train the autoencoder.

            Parameters:
                -------------
                x: array-like
                the data you wish to fit

            batch_size: int
            the batch size

            epochs: int
            number of epochs you wish to run.

            loss: string or function
            loss function. Defaults to mse

            optimizer: string or function
            optimizer. defaults to adam

            weights: string
            if weights is used, the path to the pretrained nn weights.

            verbose: int
            how verbose you wish the autoencoder to be while training.

            weight_id: string
            where you wish to save the weights

            patience: int
            if not None, the early stopping criterion
            """

        if (
            weights is None
        ):  # if there are no weights to load for the encoder, make encoder
            self.Model.compile(loss=loss, optimizer=optimizer)

            if weight_id is not None:  # if we are going to save the weights somewhere
                if patience is not None:  # if we are going to do early stopping
                    callbacks = [
                        EarlyStopping(monitor="loss", patience=patience),
                        ModelCheckpoint(
                            filepath=weight_id, monitor="loss", save_best_only=True
                        ),
                    ]
                else:
                    callbacks = [
                        ModelCheckpoint(
                            filepath=weight_id, monitor="loss", save_best_only=True
                        )
                    ]
                # fit the model with the callbacks
                self.__hist = self.Model.fit(
                    self.x_lambda(x),
                    x,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=verbose,
                )
                self.Model.save_weights(weight_id)
            else:  # if we are not saving weights
                if patience is not None:
                    callbacks = [EarlyStopping(monitor="loss", patience=patience)]
                    self.__hist = self.Model.fit(
                        self.x_lambda(x),
                        x,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=verbose,
                    )
                else:
                    self.__hist = self.Model.fit(
                        self.x_lambda(x),
                        x,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                    )
        else:  # otherwise load weights
            self.Model.load_weights(weights)

    def plot_loss(self):
        """
        """
        plt.subplots(figsize=(8, 5))
        sns.lineplot(np.arange(len(self.__hist.history['loss'])),
                     self.__hist.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')