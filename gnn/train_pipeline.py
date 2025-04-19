import torch
import torch.nn.functional as F

from gnn.model.eval import compute_tensor_accuracy


def train_gnn(model, train_data, valid_data=None, loss_fn=F.mse_loss,
              optimizer=None, epochs=10, clip_grad_norm=1.0,
              early_stop_patience=50,
              save_path=None,
              device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            ):

    model.to(device)
    if clip_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0 

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True) 
        model.train()
        optimizer.zero_grad()

        x = model(train_data.x.to(device),
                     train_data.edge_index.to(device),
                     train_data.edge_attr.to(device))
        
        pred = model.predict(x=x, edge_index=train_data.edge_index.to(device))
        
        target = train_data.edge_attr.squeeze().to(device)

        # ✅ NaN 检查
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("⚠️ NaN detected in predictions or targets!")
            return

        loss = loss_fn(pred, target)

        torch.autograd.set_detect_anomaly(True)

        loss.backward()

        optimizer.step()

        print(f"[Epoch {epoch+1:02d}] Train Loss: {loss.item():.6f}", end='')

        # ✅ 验证过程
        if valid_data:
            model.eval()
            with torch.no_grad():
                val_pred = model.predict(x=x, edge_index=valid_data.edge_index.to(device))
                val_target = valid_data.edge_attr.squeeze().to(device)

                val_loss = loss_fn(val_pred, val_target)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_state_dict = model.state_dict()
                    patience_counter = 0 
                else:
                    patience_counter += 1

                print(f" | Val Loss: {val_loss.item():.6f}")

                accuracy = compute_tensor_accuracy(val_pred, val_target, 5e-3)
                print(f" | Val Accuracy: {accuracy:.6f}")


                if patience_counter >= early_stop_patience:
                    print("🛑 Early stopping triggered.")
                    break
        else:
            print()

    # ✅ 最后加载最佳模型
    if best_state_dict:
        model.load_state_dict(best_state_dict)
        print(f"\n✅ Loaded best model with Val Loss = {best_val_loss:.6f}")
        if save_path:
            torch.save(best_state_dict, save_path)
            print(f"📦 Best model saved to: {save_path}")
    else:
        print("\n⚠️ No validation data provided or no better model found.")

    print("Best Validation Loss:", best_val_loss)

    return model
