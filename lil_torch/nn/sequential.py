from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def forward(self, input):
        for layer in self._modules.value():
            input = layer(input)

        return input

    def __getitem__(self, index):
        return list(self._modules.value())[index]
    
    def __len__(self):
        return len(self._modules)