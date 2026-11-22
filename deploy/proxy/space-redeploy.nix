# NixOS module for the proxy host: polls ghcr every 5 minutes and redeploys the
# space stack when an image digest changed. Import from configuration.nix:
#   imports = [ ./space-redeploy.nix ];
# Expects /opt/space/{check-updates.sh,redeploy.sh,docker-compose.yaml,.env}.
{ pkgs, ... }:
{
  systemd.services.space-redeploy = {
    description = "Check ghcr for new space images and redeploy";
    after = [ "network-online.target" "docker.service" ];
    wants = [ "network-online.target" ];
    path = [ pkgs.docker pkgs.docker-buildx pkgs.util-linux pkgs.gnugrep pkgs.gawk pkgs.coreutils ];
    serviceConfig = {
      Type = "oneshot";
      # check-updates.sh: 0 = no change, 10 = redeployed, 20 = error.
      SuccessExitStatus = "10";
      ExecStart = "/opt/space/check-updates.sh";
    };
  };

  systemd.timers.space-redeploy = {
    description = "Poll for new space images every 5 minutes";
    wantedBy = [ "timers.target" ];
    timerConfig = {
      OnBootSec = "2min";
      OnUnitActiveSec = "5min";
    };
  };
}
