import React, { useEffect, useState } from 'react';
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  SpeakerLayout,
  CallControls,
  StreamTheme,
} from '@stream-io/video-react-sdk';

import '@stream-io/video-react-sdk/dist/css/styles.css';

const apiKey = 'your-api-key';
const userId = 'user-id';
const token = 'your-token';
const callId = 'my-first-call';

const user = { id: userId, name: 'User Name' };

export default function App() {
  const [client, setClient] = useState();
  const [call, setCall] = useState();

  // 1. Initialize Client
  useEffect(() => {
    const _client = new StreamVideoClient({ apiKey, user, token });
    setClient(_client);

    return () => {
      _client.disconnectUser(); // Cleanup on unmount
    };
  }, []);

  // 2. Initialize Call
  useEffect(() => {
    if (!client) return;
    const _call = client.call('default', callId);
    _call.join({ create: true }); // Join or create the call
    setCall(_call);

    return () => {
      _call.leave(); // Leave call on cleanup
    };
  }, [client]);

  if (!client || !call) return <div>Initializing...</div>;

  return (
    <StreamVideo client={client}>
      <StreamTheme className="material-minimal">
        <StreamCall call={call}>
          <div className="call-container">
            <SpeakerLayout />
            <CallControls onLeave={() => console.log('Left call')} />
          </div>
        </StreamCall>
      </StreamTheme>
    </StreamVideo>
  );
}