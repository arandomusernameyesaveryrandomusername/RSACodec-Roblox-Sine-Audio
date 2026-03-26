--[[
    RSC_Fetcher.lua  —  Script (place in ServerScriptService)
    Fetches audio.rsc from GitHub on the server (HTTP is server-only),
    then fires the raw binary to each client via RemoteEvent.
]]

local HttpService  = game:GetService("HttpService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local RSC_URL = "https://github.com/arandomusernameyesaveryrandomusername/RSACodec-Roblox-Sine-Audio/raw/refs/heads/main/audio.rsc"

-- Create the RemoteEvent clients will listen on
local event = Instance.new("RemoteEvent")
event.Name   = "RSC_DataEvent"
event.Parent = ReplicatedStorage

-- Fetch once at server start
-- Append tick() as a cache-buster so every server start fetches fresh
local bustUrl = RSC_URL .. "?v=" .. tostring(math.floor(tick()))
local ok, response = pcall(function()
	return HttpService:RequestAsync({ Url = bustUrl, Method = "GET" })
end)

local rscData   = nil
local rscError  = nil

if not ok then
	rscError = "Server fetch pcall failed: " .. tostring(response)
	warn("[RSC_Fetcher]", rscError)
elseif not response.Success then
	rscError = "HTTP " .. tostring(response.StatusCode) .. " | " .. tostring(response.StatusMessage or "")
	warn("[RSC_Fetcher]", rscError)
else
	rscData = response.Body
	print(string.format("[RSC_Fetcher] Fetched %.1f KB — magic=%s",
		#rscData / 1024, string.sub(rscData, 1, 4)))
end

-- Send to each client as they join (or immediately if already connected)
game.Players.PlayerAdded:Connect(function(player)
	if rscError then
		event:FireClient(player, false, rscError)
	else
		event:FireClient(player, true, rscData)
	end
end)

-- Also fire to any players already in-game (Studio play solo)
for _, player in ipairs(game.Players:GetPlayers()) do
	if rscError then
		event:FireClient(player, false, rscError)
	else
		event:FireClient(player, true, rscData)
	end
end