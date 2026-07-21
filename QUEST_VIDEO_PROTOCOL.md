# Meta Quest Head-Camera Protocol

The Python backend controls only the RBY1 two-DOF head from the existing
`controller_state.head` HMD pose. HMD motion never changes the torso target.
The Meta Quest Unity project is not part of this repository.

## Backend setup

Install the optional streaming dependencies in the same Python environment that
runs `main.py`:

```bash
python3 -m pip install -r requirements-webrtc.txt
```

Quest video is enabled by `quest_video.enabled` in `config.yaml`. It can be
overridden with `--quest_video` or `--no_quest_video`.

## UDP discovery extension

The existing UDP message sent to the Quest app keeps `ip` and `port` and adds
an optional video descriptor:

```json
{
  "protocol_version": 2,
  "ip": "192.168.0.27",
  "port": 5005,
  "video": {
    "signaling_url": "http://192.168.0.27:8080/offer",
    "codec": "H264"
  }
}
```

When video is disabled, the `video` member is omitted. The Quest controller and
HMD pose payload sent back to UDP port 5005 is unchanged.

## WebRTC signaling

The Quest app is the receive-only offerer.

1. Create an `RTCPeerConnection` with a receive-only video transceiver.
2. Prefer H.264 and create an SDP offer.
3. POST the offer to `video.signaling_url`.
4. Apply the returned SDP answer.
5. Render the remote video texture on a head-locked 4:3 panel.

Offer request:

```json
{"type": "offer", "sdp": "<offer SDP>"}
```

Successful response:

```json
{"type": "answer", "sdp": "<answer SDP>"}
```

Only one Quest peer is supported. A new offer closes the previous peer. The
initial implementation uses LAN host ICE candidates and does not configure
STUN/TURN.

## Health endpoint

`GET http://<backend>:8080/health` returns the signaling state, selected
camera, camera running state, last frame age, codec, resolution, and FPS. A
missing or stale RealSense frame produces a black frame without stopping robot
teleoperation.

The existing notebook dashboard remains a local ZMQ client. It is not a Quest
video receiver, although its status snapshot now includes `head_tracking` and
`quest_video` diagnostics.
