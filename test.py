import tensorflow as tf

path = 'wepapp/weaponresource/checkpoints_weapon/WeaponOct24_608_8K'
model = tf.saved_model.load(path)