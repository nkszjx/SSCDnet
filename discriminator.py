from torch.autograd import Variable
import torch.nn as nn


        
class s4GAN_discriminator_DAN(nn.Module):

    def __init__(self, ndf = 64):
        super(s4GAN_discriminator_DAN, self).__init__()

        self.conv1 = nn.Conv2d(5, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20

        self.avgpool = nn.AdaptiveMaxPool2d(1) # nn.AvgPool2d((20, 20))

        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out       
        
        
        
class FCDiscriminator(nn.Module):

	def __init__(self, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(2, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x

class FCDiscriminator2(nn.Module):

    def __init__(self, num_classes, dataset, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20
        if dataset == 'pascal_voc' or dataset == 'pascal_context':
            self.avgpool = nn.AvgPool2d((20, 20))
        elif dataset == 'cityscapes':
            self.avgpool = nn.AvgPool2d((16, 32))
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps
		
		
class s4GAN_discriminator11(nn.Module):

    def __init__(self, ndf = 64):
        super(s4GAN_discriminator11, self).__init__()

        self.conv1 = nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20

        self.avgpool = nn.AdaptiveMaxPool2d(1) #nn.AvgPool2d((20, 20))

        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps

class s4GAN_discriminator(nn.Module):

    def __init__(self, num_classes, dataset, ndf = 64):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20
        if dataset == 'pascal_voc' or dataset == 'pascal_context':
            self.avgpool = nn.AvgPool2d((20, 20))
        elif dataset == 'cityscapes':
            self.avgpool = nn.AvgPool2d((16, 32))
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps

class s4GAN_discriminator_5(nn.Module):

    def __init__(self, num_classes, dataset, ndf = 64):
        super(s4GAN_discriminator_5, self).__init__()

        self.conv1 = nn.Conv2d(5, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20
        if dataset == 'pascal_voc' or dataset == 'pascal_context':
            self.avgpool = nn.AvgPool2d((20, 20))
        elif dataset == 'cityscapes':
            self.avgpool = nn.AvgPool2d((16, 32))
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps

				
		
		
class s4GAN_feature_discriminator(nn.Module):

    def __init__(self, ndf = 64):
        super(s4GAN_feature_discriminator, self).__init__()

        #self.conv1 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2	
        #self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3	
        self.conv1 = nn.Conv2d(ndf*4*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4	
		
        self.conv2 = nn.Conv2d(  ndf*4, ndf*3, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv3 = nn.Conv2d(ndf*3, ndf*2, kernel_size=4, stride=2, padding=1) # 10 x 10
        self.conv4 = nn.Conv2d(ndf*2, ndf*1, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.avgpool = nn.AvgPool2d((10, 10))	
		
        self.fc = nn.Linear(ndf*1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps
		
		

		
class Domain_adaption_discriminator(nn.Module):

    def __init__(self, ndf = 64):
        super(Domain_adaption_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1) # 160 x 160
        self.conv2 = nn.Conv2d( ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 80 x 80
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1) # 20 x 20		
        # self.avgpool = nn.AvgPool2d((20, 20))			

        # self.fc = nn.Linear(ndf*4, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
		
        x = self.conv5(x)
        x = self.leaky_relu(x)		
        #x = self.drop(x)   
        out = self.sigmoid(x)		
		
        # maps = self.avgpool(x)
        # conv4_maps = maps 
        # out = maps.view(maps.size(0), -1)
        # out = self.sigmoid(self.fc(out))
        
        return out #, conv4_maps	
		
