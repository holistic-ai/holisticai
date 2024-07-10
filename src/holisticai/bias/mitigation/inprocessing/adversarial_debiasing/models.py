import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


class ClassifierModel(nn.Module):
    features_dim: int
    hidden_size: int
    keep_prob: float

    def setup(self):
        self.encode = nn.Dense(self.hidden_size)
        self.decode = nn.Dense(1)

    @nn.compact
    def __call__(self, x, trainable=True):
        x = self.encode(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.keep_prob, deterministic=not trainable)(x)
        x = self.decode(x)
        p = nn.sigmoid(x)
        return (p, x) if trainable else p


class AdversarialModel(nn.Module):
    def setup(self):
        self.c = self.param("c", jax.random.normal, (1,))
        self.hidden = nn.Dense(1)

    @nn.compact
    def __call__(self, x, y, trainable=True):
        s = nn.sigmoid((1 + jnp.abs(self.c)) * x)
        x = jnp.concatenate([s, s * y, s * (1.0 - y)], axis=1)
        x = self.hidden(x)
        p = nn.sigmoid(x)
        return (p, x) if trainable else p


class ADModel(nn.Module):
    classifier: nn.Module
    adversarial: nn.Module

    def __call__(self, x, y, trainable=True):
        y_prob, y_logits = self.classifier(x, trainable=trainable)
        z_prob, z_logits = self.adversarial(y_logits, y, trainable=trainable)
        return y_logits, z_logits


def train_step(cls_state, adv_state, batch, use_debias, adversary_loss_weight: float, rng):
    def loss_fn(classifier_params, adversarial_params, batch, rng):
        y = batch["y"].reshape([-1, 1])
        group = batch["group"].reshape([-1, 1])
        x = batch["X"]
        rngs = {"dropout": rng}
        _, y_logits = cls_state.apply_fn({"params": classifier_params}, x, trainable=True, rngs=rngs)
        _, z_logits = adv_state.apply_fn({"params": adversarial_params}, y_logits, y, trainable=True, rngs=rngs)

        loss_cls = optax.sigmoid_binary_cross_entropy(y_logits, y).mean()
        loss_adv = optax.sigmoid_binary_cross_entropy(z_logits, group).mean()
        if use_debias:
            return loss_cls - adversary_loss_weight * loss_adv, (loss_cls, loss_adv)
        return loss_cls, (loss_cls, None)

    def adv_loss_fn(adversarial_params, classifier_params, batch, rng):
        y = batch["y"].reshape([-1, 1])
        group = batch["group"].reshape([-1, 1])
        x = batch["X"]
        rngs = {"dropout": rng}
        _, y_logits = cls_state.apply_fn({"params": classifier_params}, x, trainable=True, rngs=rngs)
        _, z_logits = adv_state.apply_fn({"params": adversarial_params}, y_logits, y, trainable=True, rngs=rngs)

        loss_adv = optax.sigmoid_binary_cross_entropy(z_logits, group).mean()
        return loss_adv

    (loss, (loss_cls, loss_adv)), grads = jax.value_and_grad(loss_fn, argnums=(0), has_aux=True)(
        cls_state.params, adv_state.params, batch, rng
    )
    new_cls_state = cls_state.apply_gradients(grads=grads)

    loss_adv, grads = jax.value_and_grad(adv_loss_fn)(adv_state.params, new_cls_state.params, batch, rng)
    new_adv_state = adv_state.apply_gradients(grads=grads)

    return new_cls_state, new_adv_state, loss_cls, loss_adv


def create_train_state(rng, model, learning_rate, feature_dim):
    adversarial_params = model.adversarial.init(rng, jnp.ones([1, 1]), jnp.ones([1, 1]))["params"]
    classifier_params = model.classifier.init(rng, jnp.ones([1, feature_dim]))["params"]
    tx = optax.sgd(learning_rate, momentum=0.9)
    adversary_tx = optax.sgd(learning_rate, momentum=0.9)
    cls_state = train_state.TrainState.create(apply_fn=model.classifier.apply, params=classifier_params, tx=tx)
    adv_state = train_state.TrainState.create(
        apply_fn=model.adversarial.apply, params=adversarial_params, tx=adversary_tx
    )
    return cls_state, adv_state
