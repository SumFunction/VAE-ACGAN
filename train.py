from torch.autograd import Variable
def reparameterize(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)
