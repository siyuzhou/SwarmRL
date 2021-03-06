import tensorflow as tf
from tensorflow.python.keras.backend import log
import tensorflow_probability as tfp
import numpy as np

import utils
from rollout_buffer import NStepRolloutBuffer


LR_A = 0.0001  # learning rate for actor
LR_C = 0.0001  # learning rate for critic

ACTOR_UPDATE_STEPS = 10  # actor update steps
CRITIC_UPDATE_STEPS = 10  # critic update steps

# ppo-clip parameters
EPSILON = 0.2


class PPOAgent:
    """
    Agent with the clipping variant of PPO method.
    """

    def __init__(self, model, action_size, action_bound=None, rollout_steps=1, memory_capacity=1e5, gamma=0.95, summary_writer=None, mode=0):
        self.model = model
        self.action_logstd = tf.Variable(-0.5 * tf.ones(action_size), name='action_logstd')

        self.rollout_buffer = NStepRolloutBuffer(
            rollout_steps, memory_capacity, gamma=gamma, num_states=2)
        self.action_bound = action_bound
        self.gamma = gamma

        self.actor_optim = tf.keras.optimizers.Adam(LR_A)
        self.critic_optim = tf.keras.optimizers.Adam(LR_C)

        self.summary_writer = summary_writer
        self.steps = 0
        self.mode = mode

    def train_actor(self, state, action, adv, old_log_prob, mask):
        if self.mode < 2:
            self.model.actor.trainable = True
        self.model.critic.trainable = False

        adv *= mask
        with tf.GradientTape() as tape:
            mean = self.model(state)[0]
            std = tf.exp(self.action_logstd)

            pi = tfp.distributions.Normal(mean, std)

            ratio = tf.exp(pi.log_prob(action) - old_log_prob)
            surr = ratio * adv

            loss = -tf.reduce_mean(
                tf.minimum(surr,
                           tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON) * adv)
            )

        trainables = self.model.trainable_variables + [self.action_logstd]
        grads = tape.gradient(loss, trainables)

        self.actor_optim.apply_gradients(zip(grads, trainables))

        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                for weights, grad in zip(trainables, grads):
                    tf.summary.histogram(weights.name.replace(':', '_'),
                                         data=weights, step=self.steps)
                    tf.summary.histogram(weights.name.replace(':', '_') +
                                         '_grads', data=grad, step=self.steps)

                # Log mean
                tf.summary.histogram('Actions/means', data=mean, step=self.steps)
                tf.summary.histogram('Actions/stds', data=std, step=self.steps)

        return loss

    def train_critic(self, state, reward, mask):
        self.model.actor.trainable = False
        self.model.critic.trainable = True

        with tf.GradientTape() as tape:
            _, values = self.model(state)
            loss = tf.keras.losses.mse(reward * mask, values * mask)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.critic_optim.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                for weights, grad in zip(self.model.trainable_variables, grads):
                    tf.summary.histogram(weights.name.replace(':', '_'),
                                         data=weights, step=self.steps)
                    tf.summary.histogram(weights.name.replace(':', '_') +
                                         '_grads', data=grad, step=self.steps)

        return loss

    def update(self, batch_size, actor_steps=ACTOR_UPDATE_STEPS, critic_steps=CRITIC_UPDATE_STEPS):
        self.steps += 1
        states, actions, rewards_to_go, old_log_probs, masks = self.rollout_buffer.get_buffer(
            batch_size)

        values = self.model(states)[1]
        adv = rewards_to_go - values  # Shape [batch, MAX_NUM_NODES, 1]

        actor_loss = 0
        for _ in range(actor_steps):
            actor_loss += self.train_actor(states, actions, adv, old_log_probs, masks)

        critic_loss = 0
        for _ in range(critic_steps):
            critic_loss += self.train_critic(states, rewards_to_go, masks)

        if self.summary_writer is not None:
            critic_loss = tf.reduce_mean(critic_loss)
            with self.summary_writer.as_default():
                if actor_steps > 0:
                    tf.summary.scalar('Actor Loss', actor_loss / actor_steps, step=self.steps)
                if critic_steps > 0:
                    tf.summary.scalar('Critic Loss', critic_loss, step=self.steps)

    def act(self, state, mask=None, training=False):
        state = utils.add_batch_dim(state)
        if mask is not None:
            mask = utils.add_batch_dim(mask)
        # state has batch dim of 1.
        action, log_prob = self.act_batch(state, mask, training)

        action, log_prob = action.numpy().squeeze(0), log_prob.numpy().squeeze(0)

        return action, log_prob

    def act_batch(self, state, mask=None, training=False):
        mean = self.model(state)[0]

        std = tf.exp(self.action_logstd)
        pi = tfp.distributions.Normal(mean, std)

        if not training:
            action = mean
        else:
            action = pi.sample()

        log_prob = pi.log_prob(action)

        if self.action_bound:
            action = tf.clip_by_value(action, -self.action_bound, self.action_bound)

        if mask is not None:
            action *= mask
            log_prob *= mask

        return action, log_prob

    def value(self, state, mask):
        state = utils.add_batch_dim(state)
        # state has batch dim of 1.
        value = self.model(state)[1]
        return value.numpy().squeeze(0) * mask

    def store_transition(self, state, action, reward, log_prob, next_state, done, mask):
        self.rollout_buffer.add_transition(state, action, reward, log_prob, mask)
        if self.rollout_buffer.path_end():
            self.finish_rollout(next_state, done, mask)

    def finish_rollout(self, next_state, done, mask):
        next_value = self.value(next_state, mask).squeeze(-1) * (1 - done)
        self.rollout_buffer.finish_path(next_value)
