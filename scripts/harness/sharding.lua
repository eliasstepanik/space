-- A tour of the demo area's four shard areas and of its far content, played
-- the way a person would play it, that asserts what a change of shard and a
-- view across shards must look like:
--
--   * at the spawn point the scene's 46 bodies are drawn, each once; the 11
--     of the far content (171 m, 706 m and 1.2 km out) at the update steps
--     of their distances (every 3rd, 6th and 30th tick), and - on a cluster
--     of several shards - as far replicas;
--   * what orbits and swings out there moves between two looks (it is not
--     frozen at its last sample), and the moons stay on their orbits;
--   * a loop round the corner where the four shard areas meet - across
--     x = -6, across z = -4, back across x = -6, back across z = -4 - changes
--     the shard four times and ends on the first one; `handoffs` counts each;
--     and in every frame of it the drawn ship is where the frame before and
--     its velocity say, within 10 cm: a change of shard is no step;
--   * flying out to the drifters, each of them goes from far replica to the
--     shard's own in place: never missing, never twice, never back, and never
--     further from where it was a frame before than it moves in that time.
--
--   space-client --headless --server HOST:PORT --script scripts/harness/sharding.lua \
--       --script-timeout 180 [--out-dir DIR]
--   scripts/dev.sh --client headless --script scripts/harness/sharding.lua
--
-- It needs the demo-area scene, a fresh client on the spawn point (so no
-- other client standing there) and a showcase nobody has knocked far out of
-- the area: physics.lua pushes a crate off for good, so run this one first
-- or on a fresh stack (the first check counts what is beyond 100 m). On the
-- default four-shard stack it checks all of the above; on one shard
-- (`--min-shards 1`) there are no borders and no far replicas, which the
-- script reads off the state - nothing is a far replica at the spawn point -
-- and then it flies the same tour and asserts that the shard never changes
-- and that everything else holds as well. It pushes nothing. Its exit code
-- is the result; the screenshots go to the output directory.

local BODIES = 46 -- the showcase's 35 and the far content's 11
local STEP_BOUND = 0.10 -- metres a drawn ship may be off its own motion, frame to frame
local DRIFT_CENTER = { 90, 40, -140 }
local FAR_ROCK = { center = { -560, 90, 420 }, orbits = { 26, 34, 42 } }
local GIANT_ROCK = { center = { -840, 320, -840 }, orbits = { 55, 70, 85 } }

local function state()
  return client.state()
end

local function sub(a, b)
  return { a[1] - b[1], a[2] - b[2], a[3] - b[3] }
end

local function len(v)
  return math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
end

local function distance(a, b)
  return len(sub(a, b))
end

local function fmt(p)
  return string.format("(%.2f %.2f %.2f)", p[1], p[2], p[3])
end

-- Turn the view at the scene point `p` (yaw left+, pitch up+, degrees).
local function aim(p)
  local s = state()
  local d = sub(p, s.scene_position)
  local yaw = math.deg(math.atan(-d[1], -d[3]))
  local pitch = math.deg(math.atan(d[2], math.sqrt(d[1] * d[1] + d[3] * d[3])))
  client.turn((yaw - s.yaw_deg + 540) % 360 - 180, pitch - s.pitch_deg)
end

-- Turn the view towards the horizontal direction (dx, dz), level.
local function face(dx, dz)
  local s = state()
  local want = math.deg(math.atan(-dx, -dz))
  local turn = (want - s.yaw_deg + 540) % 360 - 180
  if math.abs(turn) > 0.5 or math.abs(s.pitch_deg) > 0.5 then
    client.turn(turn, -s.pitch_deg)
  end
end

local held = {}
local function hold(key, down)
  if down and not held[key] then
    client.key_down(key)
    held[key] = true
  elseif not down and held[key] then
    client.key_up(key)
    held[key] = false
  end
end

local function let_go()
  for _, key in ipairs({ "w", "s", "a", "d", "space", "shift" }) do
    hold(key, false)
  end
end

local function steer(speed, want, more, less, slack)
  hold(more, speed < want - slack)
  hold(less, speed > want + slack)
end

-- What every frame of the tour must show ------------------------------------

local watch = {
  last = nil, -- the last frame's drawn ship
  worst = 0, -- the largest step off the ship's own motion, metres
  worst_at_switch = 0, -- the same within half a second of a change of shard
  frames = 0,
  switches = {}, -- { time, from, to }
  shard = nil,
  recent = {}, -- the last frames' { time, step }
  bodies = {}, -- guid -> { source, position, time } of the bodies followed
  follow = {}, -- guid -> true: the bodies whose way from far to local is followed
  body_worst = 0,
}

local rank = { far = 1, ghost = 2, ["local"] = 3 }

local function look()
  local s = state()
  if s.drawn_position == nil or s.drawn_tick == nil then
    return s
  end
  local now = client.time()
  local last = watch.last
  if last ~= nil and s.frame ~= last.frame then
    watch.frames = watch.frames + 1
    -- One timeline: the tick the ship is drawn at never goes back.
    assert(s.drawn_tick >= last.tick,
      string.format("the drawn tick went back from %.2f to %.2f", last.tick, s.drawn_tick))
    local dt = (s.drawn_tick - last.tick) / 30
    -- Where the frame before and the velocities of the two say it is.
    local step = 0
    if dt < 0.5 then
      local expected = {}
      for k = 1, 3 do
        expected[k] = last.position[k] + 0.5 * (last.velocity[k] + s.drawn_velocity[k]) * dt
      end
      step = distance(s.drawn_position, expected)
    end
    assert(step <= STEP_BOUND, string.format(
      "the drawn ship stepped %.3f m off its own motion between two frames at %s (shard %s -> %s, %.1f m/s)",
      step, fmt(s.drawn_position), tostring(last.shard), tostring(s.shard), s.speed))
    if step > watch.worst then watch.worst = step end
    table.insert(watch.recent, { now, step })
    while #watch.recent > 0 and watch.recent[1][1] < now - 0.5 do
      table.remove(watch.recent, 1)
    end
    if s.shard ~= last.shard then
      table.insert(watch.switches, { now, last.shard, s.shard })
      watch.since_switch = now
      for _, r in ipairs(watch.recent) do
        if r[2] > watch.worst_at_switch then watch.worst_at_switch = r[2] end
      end
    elseif watch.since_switch ~= nil and now - watch.since_switch < 0.5 then
      if step > watch.worst_at_switch then watch.worst_at_switch = step end
    end
  end
  watch.last = {
    frame = s.frame, tick = s.drawn_tick, position = s.drawn_position,
    velocity = s.drawn_velocity, shard = s.shard,
  }
  -- Every body, once, in every frame.
  local seen = {}
  for _, b in ipairs(s.bodies) do
    assert(not seen[b.guid], "the body " .. b.guid .. " is drawn twice")
    seen[b.guid] = true
    if watch.follow[b.guid] then
      local was = watch.bodies[b.guid]
      if was ~= nil then
        assert(rank[b.source] >= rank[was.source], string.format(
          "%s went back from %s to %s", b.guid, was.source, b.source))
        -- It is where it was, moved on by no more than it moves.
        local moved = distance(b.position, was.position)
        local allowed = (math.max(b.speed, was.speed) + 0.5) * (now - was.time) + 0.1
        assert(moved <= allowed, string.format(
          "%s jumped %.2f m between two frames (%s -> %s, %.2f m/s)",
          b.guid, moved, was.source, b.source, b.speed))
        if moved > watch.body_worst then watch.body_worst = moved end
        if b.source ~= was.source then
          client.log(string.format("  %s: %s -> %s at %.0f m (step %d)",
            b.guid, was.source, b.source, b.distance, b.step))
        end
      end
      watch.bodies[b.guid] = {
        source = b.source, position = b.position, speed = b.speed, time = now,
      }
    end
  end
  assert(#s.bodies == BODIES, string.format(
    "%d bodies are drawn at %s on shard %s, not %d", #s.bodies, fmt(s.scene_position),
    tostring(s.shard), BODIES))
  return s
end

-- Fly to `target` (metres from the spawn point) as a pilot would with the
-- keys - face it, thrust towards the speed that gets there and stops, at
-- most `cruise` m/s - looking at every frame on the way. Weightless space
-- only. Returns when within `tolerance` and slow.
local function fly_to(target, tolerance, timeout, cruise)
  local started = client.time()
  while true do
    local s = look()
    local p, v = s.scene_position, s.velocity
    local dx, dy, dz = target[1] - p[1], target[2] - p[2], target[3] - p[3]
    local flat = math.sqrt(dx * dx + dz * dz)
    if flat > 0.4 then
      face(dx, dz)
    end
    local yaw = math.rad(state().yaw_deg)
    local ahead = -math.sin(yaw) * v[1] - math.cos(yaw) * v[3]
    local across = math.cos(yaw) * v[1] - math.sin(yaw) * v[3]
    local towards = math.min(cruise, 0.7 * math.sqrt(4 * math.max(flat - 0.1, 0)))
    local rise = math.max(-cruise, math.min(cruise, 1.5 * dy))
    steer(v[2], rise, "space", "shift", 0.2)
    steer(ahead, towards, "w", "s", 0.25)
    steer(across, 0, "d", "a", 0.25)
    if flat <= tolerance and math.abs(dy) <= tolerance + 0.3 and s.speed < 0.6 then
      let_go()
      return s
    end
    if client.time() - started > timeout then
      let_go()
      error(string.format("could not reach %s in %d s: at %s, %.1f m/s",
        fmt(target), timeout, fmt(p), s.speed))
    end
    client.wait_frames(1)
  end
end

-- Keep looking for `seconds`.
local function watch_for(seconds)
  local until_ = client.time() + seconds
  while client.time() < until_ do
    look()
    client.wait_frames(1)
  end
end

-- 1. Online, on the spawn point, the whole scene drawn -----------------------

client.wait_frames(1)
client.wait_until(function() return state().screen ~= "connecting" end, 20)
if state().screen ~= "online" then
  client.connect(state().address, 20)
end
client.wait_until(function()
  local s = state()
  assert(s.screen ~= "error", "cannot connect: " .. tostring(s.error))
  return s.screen == "online" and s.scene_position ~= nil and s.drawn_position ~= nil
end, 20)
local s = state()
assert(s.scene == "demo-area", "the shard shows " .. tostring(s.scene) .. ", not the demo area")
assert(distance(s.scene_position, { 0, 0, 0 }) < 0.5,
  "a new player starts on the spawn point, not at " .. fmt(s.scene_position))
client.wait_until(function()
  local now = state()
  if #now.bodies ~= BODIES then return false end
  -- (The steps settle within a second of the first frame.)
  local coarse = 0
  for _, b in ipairs(now.bodies) do
    if b.step > 0 then coarse = coarse + 1 end
  end
  return coarse == 11
end, 15)
s = state()
assert(#s.bodies == BODIES, string.format("%d bodies are drawn, not %d", #s.bodies, BODIES))

-- The far content: what is further than 100 m. Five drifters at step 1,
-- three moons at step 2, three at step 3 - on any number of shards.
local far, by_step = {}, { 0, 0, 0 }
local far_replicas, others_far = 0, 0
for _, b in ipairs(s.bodies) do
  if b.distance > 100 then
    table.insert(far, b)
    assert(b.step >= 1 and b.step <= 3,
      string.format("%s at %.0f m is at step %d", b.guid, b.distance, b.step))
    by_step[b.step] = by_step[b.step] + 1
    local want = b.distance < 250 and 1 or (b.distance < 1000 and 2 or 3)
    assert(b.step == want, string.format("%s %s at %.0f m is sent at step %d, not %d",
      b.visual, b.guid, b.distance, b.step, want))
    if b.source == "far" then far_replicas = far_replicas + 1 end
  else
    assert(b.step == 0, string.format("%s at %.0f m is at step %d", b.guid, b.distance, b.step))
    if b.source == "far" then others_far = others_far + 1 end
  end
end
assert(#far == 11, string.format("%d bodies are further than 100 m, not the far content's 11", #far))
assert(by_step[1] == 5 and by_step[2] == 3 and by_step[3] == 3, string.format(
  "the far content's steps: %d at 10 Hz, %d at 5 Hz, %d at 1 Hz (5, 3, 3 expected)",
  by_step[1], by_step[2], by_step[3]))
assert(others_far == 0, "a body of the showcase is a far replica at the spawn point")
-- Several shards, or one? Nothing else in the script is told which.
local sharded = far_replicas > 0
if sharded then
  assert(far_replicas == 11, string.format(
    "%d of the far content's 11 bodies are far replicas at the spawn point", far_replicas))
  assert(s.far_drawn == 11, "far_drawn is " .. tostring(s.far_drawn))
else
  for _, b in ipairs(s.bodies) do
    assert(b.source == "local", b.guid .. " is a " .. b.source .. " on a cluster of one shard")
  end
end
client.log(string.format("at the spawn point on shard %s: %d bodies, %d far replicas -> %s",
  tostring(s.shard), #s.bodies, far_replicas,
  sharded and "several shards" or "one shard: no borders to cross"))
client.screenshot("sharding-1-spawn.jpg")

-- 2. The far content, from here ----------------------------------------------

-- First person: nothing of the own character in the way.
client.type("c")
aim(DRIFT_CENTER)
client.wait_frames(3)
client.screenshot("sharding-2-drifters.jpg")
aim(FAR_ROCK.center)
client.wait_frames(3)
client.screenshot("sharding-3-far-rock.jpg")
aim(GIANT_ROCK.center)
client.wait_frames(3)
client.screenshot("sharding-4-giant-rock.jpg")
client.type("c")

-- What moves out there moves between two looks: three looks 1.5 s apart.
local function moons_of(rock, bodies)
  local out = {}
  for _, b in ipairs(bodies) do
    local r = distance(b.position, rock.center)
    if b.visual == "ball" and r < rock.orbits[3] + 10 then
      out[b.guid] = { position = b.position, radius = r }
    end
  end
  return out
end
local function count(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  return n
end
local before = { moons_of(FAR_ROCK, s.bodies), moons_of(GIANT_ROCK, s.bodies) }
assert(count(before[1]) == 3 and count(before[2]) == 3, "three moons round each rock")
local pair_before = nil
for round = 1, 3 do
  client.wait(1.5)
  local now = state()
  for i, rock in ipairs({ FAR_ROCK, GIANT_ROCK }) do
    local moons = moons_of(rock, now.bodies)
    for guid, was in pairs(before[i]) do
      local is = moons[guid]
      assert(is ~= nil, "the moon " .. guid .. " is gone")
      local moved = distance(is.position, was.position)
      -- (They fly at 5 to 9 m/s: 7 to 14 m in the 1.5 s.)
      assert(moved > 3 and moved < 25, string.format(
        "look %d: the moon %s moved %.2f m in 1.5 s", round, guid, moved))
      local on = false
      for _, orbit in ipairs(rock.orbits) do
        if math.abs(is.radius - orbit) < 0.03 * orbit then on = true end
      end
      assert(on, string.format("the moon %s is %.1f m from its rock: on none of its orbits",
        guid, is.radius))
    end
    before[i] = moons
  end
  -- The spring pair: the two drifters that move.
  local pair = {}
  for _, b in ipairs(now.bodies) do
    if b.visual == "crate" and distance(b.position, DRIFT_CENTER) < 12
      and math.abs(b.position[2] - DRIFT_CENTER[2]) < 0.5 then
      table.insert(pair, b)
    end
  end
  assert(#pair == 2, "the drifters' spring pair: " .. #pair .. " crates")
  local apart = distance(pair[1].position, pair[2].position)
  assert(apart > 11.5 and apart < 20.5, string.format("the pair is %.2f m apart", apart))
  if pair_before ~= nil then
    assert(math.abs(apart - pair_before) > 0.2, string.format(
      "look %d: the spring pair did not move (%.2f m apart, as before)", round, apart))
  end
  pair_before = apart
end
client.log("the moons and the spring pair moved between three looks 1.5 s apart")

-- 3. Round the corner where the four shard areas meet -------------------------

local first_shard = s.shard
local first_handoffs = s.handoffs
-- Up over everything that stands or floats in the area, then the loop: each
-- leg ends 8 m past a border, and is flown on while the shard changes (a
-- metre past the border and about a second later).
fly_to({ 0, 6, 2 }, 0.5, 20, 3)
local legs = {
  { { -14, 6, 2 }, "x = -6" },
  { { -14, 6, -12 }, "z = -4" },
  { { 2, 6, -12 }, "x = -6, back" },
  { { 2, 6, 4 }, "z = -4, back" },
}
local shards_seen = { [tostring(first_shard)] = true }
for i, leg in ipairs(legs) do
  local from = state().shard
  fly_to(leg[1], 0.6, 30, 4)
  if sharded then
    client.wait_until(function() return look().shard ~= from end, 6)
  else
    watch_for(1.5)
  end
  local now = look()
  if sharded then
    assert(now.shard ~= from, "across " .. leg[2] .. " the shard is still " .. tostring(from))
    assert(now.handoffs == first_handoffs + i, string.format(
      "across %s: handoffs is %d, not %d", leg[2], now.handoffs, first_handoffs + i))
    shards_seen[tostring(now.shard)] = true
    client.log(string.format("across %s: shard %s -> %s (%s)", leg[2], tostring(from),
      tostring(now.shard), tostring(now.shard_address)))
  else
    assert(now.shard == first_shard and now.handoffs == first_handoffs,
      "the shard changed on a cluster of one shard")
  end
  assert(now.address == s.address, "the entry address changed")
  client.screenshot(string.format("sharding-5-leg-%d.jpg", i))
end
s = look()
if sharded then
  assert(s.shard == first_shard, string.format(
    "back in the spawn point's cell on shard %s, not %s", tostring(s.shard), tostring(first_shard)))
  assert(count(shards_seen) == 4, "the loop went through " .. count(shards_seen) .. " shards, not 4")
  assert(#watch.switches == 4, #watch.switches .. " changes of shard were seen in the frames")
end
client.log(string.format(
  "the loop: %d frames, %d changes of shard; the drawn ship was at most %.1f mm off its own motion, %.1f mm within 0.5 s of a change",
  watch.frames, #watch.switches, watch.worst * 1000, watch.worst_at_switch * 1000))

-- 4. Out to the drifters: far replica -> the shard's own ----------------------

local drifters = {}
for _, b in ipairs(s.bodies) do
  if b.visual == "crate" and distance(b.position, DRIFT_CENTER) < 25 then
    table.insert(drifters, b)
    watch.follow[b.guid] = true
    if sharded then
      assert(b.source == "far" and b.step == 1,
        string.format("a drifter is %s at step %d from the spawn point's cell", b.source, b.step))
    end
  end
end
assert(#drifters == 5, #drifters .. " drifters")
local handoffs = s.handoffs
-- 45 m short of their middle, on the line from the spawn point.
fly_to({ 66, 30, -102 }, 1.5, 60, 20)
watch_for(1.0)
s = look()
for _, b in ipairs(s.bodies) do
  if watch.follow[b.guid] then
    assert(b.source == "local", "a drifter is a " .. b.source .. " from 45 m away")
    assert(b.step == 0, "a drifter 45 m away is sent at step " .. b.step)
    assert(b.distance < 70, string.format("a drifter is %.0f m away", b.distance))
  end
end
if sharded then
  assert(s.handoffs == handoffs + 1 and s.shard ~= first_shard,
    "flying out to the drifters is one change of shard")
end
-- From here the six moons of the two rocks are still another shard's, far
-- off: far replicas on several shards, the shard's own on one.
local moons = 0
for _, b in ipairs(s.bodies) do
  if b.visual == "ball" and (distance(b.position, FAR_ROCK.center) < 60
      or distance(b.position, GIANT_ROCK.center) < 110) then
    moons = moons + 1
    assert(b.source == (sharded and "far" or "local"),
      "a moon of a far rock is a " .. b.source .. " from the drifters")
  end
end
assert(moons == 6, moons .. " moons round the two rocks")
client.type("c")
aim(DRIFT_CENTER)
client.wait_frames(3)
client.screenshot("sharding-6-at-the-drifters.jpg")
client.type("c")
client.log(string.format(
  "at the drifters on shard %s: 46 bodies in every one of %d frames; no drifter moved more than %.2f m between two frames",
  tostring(s.shard), watch.frames, watch.body_worst))
client.log(string.format("largest step of the drawn ship off its own motion: %.1f mm (bound %.0f mm)",
  watch.worst * 1000, STEP_BOUND * 1000))
