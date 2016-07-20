local Rack, parent = torch.class('nn.Rack', 'nn.Module')

function Rack:__init(alpha)
   parent.__init(self)
   self.alpha = alpha or 1
end

function Rack:updateOutput(input)
   input.THNN.Rack_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.alpha
   )
   return self.output
end

function Rack:updateGradInput(input, gradOutput)
   input.THNN.Rack_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.alpha
   )
   return self.gradInput
end
