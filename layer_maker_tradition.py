import tensorflow as tf
import config as cfg
import numpy as np
class layermaker():
    def __init__(self,truncated_mean,truncated_stddev,bias,name_scope):
        self.name_scope = name_scope
        self.truncated_mean = truncated_mean
        self.truncated_stddev = truncated_stddev
        self.bias = bias
        self.idx_conv = 1
        self.idx_el = 1
        self.variable_ls = []
    def conv_layer(self,input,size,in_chanel,out_chanel,stride = 1,padding = 'SAME',linear = False,l2 = False):
        kernel_init = tf.Variable(tf.truncated_normal([size,size,in_chanel,out_chanel],mean=self.truncated_mean,stddev=self.truncated_stddev,dtype=tf.float32), name=self.name_scope+'conv_kernel'+str(self.idx_conv))
        conv = tf.nn.conv2d(input,kernel_init,strides=[1,stride,stride,1],padding=padding)
        bias_init = tf.Variable(tf.constant(self.bias, shape=[out_chanel],dtype=tf.float32), name=self.name_scope+'bias'+str(self.idx_conv))
        conv_bias = conv+bias_init
        self.idx_conv = self.idx_conv + 1
        self.variable_ls.append(kernel_init)
        self.variable_ls.append(bias_init)
        if(linear):
            if(l2):
                return conv_bias,kernel_init,bias_init
            else:
                return conv_bias
        else:
            if (l2):
                return tf.maximum(0.1 * conv_bias, conv_bias), kernel_init, bias_init
            else:
                return tf.maximum(0.1 * conv_bias, conv_bias)


    def pool_layer(self, input, size = 2, stride = 2):
        return tf.nn.max_pool(input, ksize=[1,size,size,1], strides=[1,stride,stride,1],padding='VALID')

    def aver_pool_layer(self, input, size=2, stride=2):
        return tf.nn.avg_pool(input, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')

    def bilinear_upsample_layer(self,input,in_chanel,out_size):
        upsample_kernel = np.array([[0.0625, 0.1875, 0.1875, 0.0625], [0.1875, 0.5625, 0.5625, 0.1875],
                                    [0.1875, 0.5625, 0.5625, 0.1875], [0.0625, 0.1875, 0.1875, 0.0625]],
                                   dtype=np.float32)
        upsample_filter_np = np.zeros((4, 4, in_chanel, in_chanel), dtype=np.float32)
        for i in range(in_chanel):
            upsample_filter_np[:, :, i, i] = upsample_kernel

        upsample_filter_tf = tf.constant(upsample_filter_np,dtype=tf.float32)

        out = tf.nn.conv2d_transpose(input, upsample_filter_tf,
                                     output_shape=[cfg.BATCH_SIZE, out_size, out_size, in_chanel],
                                     strides=[1, 2, 2, 1])
        return out

    def deconv_layer(self,input, scale, out_size, in_chanel, out_chanel,linear = False,l2=False,stride = 2):
        kernel_init = tf.Variable(
            tf.truncated_normal([scale, scale, out_chanel, in_chanel], mean=self.truncated_mean,
                                stddev=self.truncated_stddev,
                                dtype=tf.float32), name=self.name_scope + 'deconv_kernel' + str(self.idx_conv))
        bias_init = tf.Variable(tf.constant(self.bias, shape=[out_chanel], dtype=tf.float32),
                                name=self.name_scope + 'bias' + str(self.idx_conv))
        deconv = tf.nn.conv2d_transpose(input, kernel_init, [cfg.BATCH_SIZE, out_size, out_size, out_chanel],
                                     strides=[1, stride, stride, 1],padding='VALID')
        deconv_bias = deconv+bias_init
        self.idx_conv = self.idx_conv + 1

        if (linear):
            if (l2):
                return deconv_bias, kernel_init, bias_init
            else:
                return deconv_bias
        else:
            if (l2):
                return tf.maximum(0.1 * deconv_bias, deconv_bias), kernel_init, bias_init
            else:
                return tf.maximum(0.1 * deconv_bias, deconv_bias)

    def slice_layer(self,input,out_size,chanel,edge_cut=1):
        out = tf.slice(input,[0,edge_cut,edge_cut,0],[cfg.BATCH_SIZE,out_size,out_size,chanel])
        return out

