import tensorflow as tf
import tensorflow_probability as tfp

from rollout_buffer import RolloutBuffer


LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic

ACTOR_UPDATE_STEPS = 10  # actor update steps
CRITIC_UPDATE_STEPS = 10  # critic update steps

# ppo-clip parameters
EPSILON = 0.2


class PPOAgent:
    """
    Agent with the clipping variant of PPO method.
    """

    def __init__(self, model, action_size, action_bound, summary_writer):
        self.model = model
        self.action_std = tf.Variable(tf.ones(action_size), name='action_std')

        self.rollout_buffer = RolloutBuffer(num_state_inputs=2)
        self.action_bound = action_bound

        self.actor_optim = tf.keras.optimizers.Adam(LR_A)
        self.critic_optim = tf.keras.optimizers.Adam(LR_C)

        self.summary_writer = summary_writer
        self.steps = 0

    def train_actor(self, state, action, adv, old_pi):
        self.model.actor.trainable = True
        self.model.critic.trainable = False

        with tf.GradientTape() as tape:
            mean, _ = self.model(state)
            std = self.action_std

            pi = tfp.distributions.Normal(mean, std)

            ratio = pi.prob(action) / old_pi.prob(action)
            surr = ratio * adv

            loss = -tf.reduce_mean(
                tf.minimum(surr,
                           tf.clip_by_value(ratio, 1-EPSILON, 1+EPSILON) * adv)
            )

        trainables = self.model.trainable_variables + [self.action_std]
        grads = tape.gradient(loss, trainables)
        self.actor_optim.apply_gradients(zip(grads, trainables))

        with self.summary_writer.as_default():
            for weights, grad in zip(self.model.actor.trainable_weights, grads):
                tf.summary.histogram(weights.name.replace(':', '_') + '_grads', data=grad, step=self.steps)

        return loss

    def train_critic(self, state, reward):
        self.model.actor.trainable = False
        self.model.critic.trainable = True

        with tf.GradientTape() as tape:
            _, values = self.model(state)
            loss = tf.keras.losses.mse(reward, values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.critic_optim.apply_gradients(zip(grads, self.model.trainable_variables))

        with self.summary_writer.as_default():
            for weights, grad in zip(self.model.critic.trainable_weights, grads):
                tf.summary.histogram(weights.name.replace(':', '_') + '_grads', data=grad, step=self.steps)

        return loss

    def update(self, actor_steps=ACTOR_UPDATE_STEPS, critic_steps=CRITIC_UPDATE_STEPS):
        states, actions, rewards_to_go = self.rollout_buffer.get_buffer()
        if states:
            mean, values = self.model(states)
            std = self.action_std
            pi = tfp.distributions.Normal(mean, std)
            adv = rewards_to_go - values

            actor_loss = 0
            critic_loss = 0
            for _ in range(actor_steps):
                actor_loss += self.train_actor(states, actions, adv, pi)

            actor_loss /= actor_steps

            for _ in range(critic_steps):
                critic_loss += self.train_critic(states, rewards_to_go)

            critic_loss /= critic_steps

            with self.summary_writer.as_default():
                tf.summary.scalar('Actor Loss', actor_loss, step=self.steps)
                tf.summary.scalar('Critic Loss', critic_loss, step=self.steps)

        self.rollout_buffer.clear()

    def act(self, state, greedy=False):
        self.steps += 1
        # state has batch dim of 1.
        mean, _ = self.model(state)
        if greedy:
            action = mean
        else:
            std = self.action_std
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()

        return tf.clip_by_value(action, -self.action_bound, self.action_bound).numpy().squeeze()

    def value(self, state):
        # state has batch dim of 1.
        _, value = self.model(state)
        return value.numpy().squeeze()

    def store_transition(self, state, action, reward):
        self.rollout_buffer.add_transition(state, action, reward)

    def finish_rollout(self, next_state, done):
        next_state_value = self.value(next_state) * (1 - done)
        self.rollout_buffer.finish_path(next_state_value)
