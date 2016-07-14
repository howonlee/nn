local HarmLU, parent = torch.class('nn.HarmLU', 'nn.Module')

--[[
--Do you even power law, bro?
--]]

function HarmLU:__init(alpha, inplace)
   parent.__init(self)
   self.alpha = alpha or 1
   assert(type(self.alpha) == 'number')
   self.inplace = inplace or false
   assert(type(self.inplace) == 'boolean')
end

function HarmLU:updateOutput(input)
   local inplace = self.inplace or false

   input.THNN.HarmLU_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.alpha,
      inplace
   )
   return self.output
end

function HarmLU:updateGradInput(input, gradOutput)
   local inplace = self.inplace or false

   input.THNN.HarmLU_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata(),
      self.alpha,
      inplace
   )
   return self.gradInput
end

function HarmLU:__tostring__()
  return string.format('%s (alpha:%f)', torch.type(self), self.alpha)
end
