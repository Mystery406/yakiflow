-- Reload the selected external subtitle track whenever its file changes.
--
-- YakiFlow rewrites the live subtitle file atomically while a stream is
-- running, so every modification-time change is a complete new version.
-- Load this script with ``mpv --script=.../sub-autoreload.lua`` or copy it
-- into mpv's scripts directory (e.g. ``~/.config/mpv/scripts/``); it does
-- nothing for subtitle files that never change.
--
-- The poll interval can be adjusted with
-- ``--script-opts=sub-autoreload-interval=<seconds>``.

local utils = require "mp.utils"
local options = require "mp.options"

local opts = { interval = 2 }
options.read_options(opts, "sub-autoreload")

local watched_path = nil
local watched_mtime = nil
local watched_size = nil

mp.add_periodic_timer(opts.interval, function()
    for _, track in ipairs(mp.get_property_native("track-list") or {}) do
        if track.type == "sub" and track.selected and track.external then
            local path = track["external-filename"]
            local info = path and utils.file_info(path)
            if not info then
                return
            end
            if path ~= watched_path then
                -- A newly selected file was just loaded by mpv itself.
                watched_path = path
                watched_mtime = info.mtime
                watched_size = info.size
            elseif info.mtime ~= watched_mtime or info.size ~= watched_size then
                watched_mtime = info.mtime
                watched_size = info.size
                mp.commandv("sub-reload", tostring(track.id))
            end
            return
        end
    end
end)
