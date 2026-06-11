import { CheckCircle2, ServerCrash, Timer } from "lucide-react";

const iconByState = {
  online: CheckCircle2,
  offline: ServerCrash,
  checking: Timer
};

export function BackendStatus({ status }) {
  const Icon = iconByState[status.state] || Timer;

  return (
    <section className={`backend-status ${status.state}`} aria-label="Backend status">
      <Icon aria-hidden="true" size={18} />
      <div>
        <strong>{status.title}</strong>
        <span>{status.detail}</span>
      </div>
    </section>
  );
}
