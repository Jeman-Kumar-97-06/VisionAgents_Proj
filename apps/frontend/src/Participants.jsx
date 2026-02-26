import {
  CallParticipantsList,
  StreamCall,
  useCall
} from "@stream-io/video-react-sdk";

const MyParticipantList = () => {
  const call = useCall();
  return (
    <StreamCall call={call}>
      <CallParticipantsList />
    </StreamCall>
  );
};

export default MyParticipantList;