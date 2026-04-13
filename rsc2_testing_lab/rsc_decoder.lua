--!native
-- Simplified RSC2 Fetcher + Sound Pool Builder

local SoundService      = game:GetService("SoundService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local tcreate           = table.create

local SINE_ASSET_ID = "rbxassetid://113823829598029"

-- ══════════════════════════════════════════════════
-- SOUND POOL BUILDER
-- ══════════════════════════════════════════════════
local function buildSoundPool(nSlots)
	local pool      = tcreate(nSlots)
	local container = Instance.new("Folder")
	container.Name   = "RSC2_SoundPool"
	container.Parent = SoundService

	for i = 1, nSlots do
		local s = Instance.new("Sound")
		s.SoundId            = SINE_ASSET_ID
		s.Volume             = 0
		s.Looped             = true
		s.RollOffMaxDistance = 0
		s.Parent             = container
		pool[i] = s
	end

	-- Wait for asset to load on the first sound instance
	if not pool[1].IsLoaded then
		local loaded = false
		local connection
		connection = pool[1].Loaded:Connect(function() 
			loaded = true 
			connection:Disconnect()
		end)
		
		local t = 0
		while not loaded and t < 10 do
			task.wait(0.1)
			t += 0.1
		end
	end

	-- Start all sounds in the pool (muted)
	for i = 1, nSlots do 
		pool[i]:Play() 
	end

	return pool, container
end

-- ══════════════════════════════════════════════════
-- FETCH RSC DATA
-- ══════════════════════════════════════════════════
coroutine.wrap(function()
	print("[RSC2] Waiting for server...")
	local event = ReplicatedStorage:WaitForChild("RSC_DataEvent", 30)
	
	if not event then
		warn("[RSC2] RSC_DataEvent not found")
		return
	end

	local fetchOk, raw
	local connection
	connection = event.OnClientEvent:Connect(function(ok, payload)
		fetchOk = ok
		raw     = payload
		connection:Disconnect()
	end)

	-- Wait for data to arrive
	local t = 0
	while raw == nil and t < 30 do 
		task.wait(0.1)
		t += 0.1 
	end

	if raw == nil then
		warn("[RSC2] Timed out waiting for data")
		return
	end

	if not fetchOk then
		warn("[RSC2] Server fetch failed:", raw)
		return
	end

	print("[RSC2] Received", #raw, "bytes")
	
	-- Example usage:
	-- local pool, container = buildSoundPool(64) 
	-- print("Sound pool initialized with 64 slots.")
end)()